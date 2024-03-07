from flask import render_template, request, redirect, url_for, flash, Flask, jsonify, session, make_response
from models import User, Profile, Cart, Product, Order, Counter, Flavors, Shipping
from mongoengine import connect
from flask_restful import Api
from flask_mongoengine import MongoEngine
import routes
from itsdangerous import URLSafeTimedSerializer, BadSignature
from flask_mail import Mail
from flask.json import JSONEncoder
from json import JSONEncoder
import os
import random
from bson.objectid import ObjectId





app = Flask(__name__)
app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost:27017/cakemarque'
}

app.config['SECRET_KEY'] = '1234asdf#$%&hjkl'

# instantiate mongo engine
db=MongoEngine()
db.init_app(app)

api = Api(app)
# initialize routes
routes.initialize_routes(api)


# mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Change to your email provider's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'youremailusername'
app.config['MAIL_PASSWORD'] = 'youremailpassword'
app.config['MAIL_DEFAULT_SENDER'] = 'youremail@gmail.com'

mail = Mail(app)


@app.before_first_request
def initialize_counters():
    # Check if the counter document already exists
    counter = Counter.objects(name="product_counter").first()

    if counter is None:
        # If it doesn't exist, create it
        Counter(name="product_counter").save()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('log.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    user_id = session.get('user_id')
    if user_id:
        user = User.objects(id=user_id).first()
        if user and str(user.role) == 'customer':
            return render_template('customer-dashboard.html')
        if user and str(user.role) == 'admin':
            return render_template('admin-dashboard.html')
    else:     
        return render_template('log.html')
@app.route('/customer-dashboard')
def customer_dashboard():
    user_id = session.get('user_id')
    if user_id:
        user = User.objects(id=user_id).first()
        if user and str(user.role) == 'customer':
            return render_template('customer-dashboard.html')
    # If not logged in or not a customer, redirect to login page
    return redirect(url_for('login'))

@app.route('/admin-dashboard')
def admin_dashboard():
    user_id = session.get('user_id')
    if user_id:
        user = User.objects(id=user_id).first()
        if user and str(user.role) == 'admin':
            return render_template('admin-dashboard.html')
    # If not logged in or not an admin, redirect to login page
    return redirect(url_for('login'))

@app.route('/verify-email')
def verify_email():
    return render_template('verify-account.html')

@app.route('/insert-product')
def insert_product():
    user_id = session.get('user_id')
    if user_id:
        user = User.objects(id=user_id).first()
        if user and str(user.role) == 'admin':
            return render_template('insert-product.html')
    return redirect(url_for('login'))
@app.route('/shop')
def shop_page():
    return render_template('shop.html')

@app.route('/gallery-viewer')
def gallery_viewer():
    return render_template('gallery-viewer.html')

def get_product_by_id(product_id):
    # Query the database for the product with the given ID
    product = Product.objects(product_id=product_id).first()

    if product:
        start_index = product.product_image.index('/images/')
        substring = product.product_image[start_index:]
        product_data = {
            'product_id':product_id,
            'product_image': substring,
            'product_name': product.product_name,
            'short_description': product.short_description,
            'long_description': product.long_description,
            'constant_price': product.constant_price,
            'minimum_size': product.minimum_size,

            # Add other fields as needed
        }
        # print("Product Data:",product_data)
        
        return product_data
        # Return the product data as JSON response
    else:
        # Product not found, return an error message
        return jsonify({'error': 'Product not found'}), 404


@app.route('/api/products')
def get_products():
    products = Product.objects.all()
    product_data = []
    for product in products:
        # print("Price: ",product.constant_price)
        product_data.append({
            '_id': str(product._id),
            'product_id': product.product_id,
            'product_image': product.product_image,
            'product_name': product.product_name,
            'constant_price': product.constant_price,
            'categories':product.categories,
            # Add other product fields as needed
        })
    return jsonify(product_data)

@app.route('/api/delivery-charges')
def get_delivery_charges():
    entries = Shipping.objects.all()
    shipping_data = []
    for entry in entries:
        shipping_data.append({
            'area': entry.area,
            'charges': entry.charges,
        })
    return jsonify(shipping_data)

@app.route('/products/<product_id>')
def product_detail(product_id):
    product = get_product_by_id(product_id)  # Implement this function

    # Render the product detail page with the product information
    response = make_response(render_template('product-detail.html', product=product))
    
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'

    return response

@app.route('/get_flavors')
def get_flavors():
    flavors = Flavors.objects()
    flavors_list = list(flavors)
    return jsonify(flavors_list)

@app.route('/get_shipping')
def get_shipping():
    shipping = Shipping.objects()
    shipping_list = list(shipping)
    return jsonify(shipping_list)

@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/carts')
def carts():
    return render_template('carts.html')

@app.route('/checkout/<website_id>/<cart_name>', methods=['GET', 'POST'])
def guest_checkout(website_id, cart_name):
    cart_session_key = f'cart_{website_id}_{cart_name}'
    if request.method == 'POST':
        # Create a dictionary to store checkout information
        checkout_info = {}

        # Get the form data from the POST request
        checkout_info['name'] = request.form['name']
        checkout_info['email'] = request.form['email']
        checkout_info['phone'] = request.form['phone']
        checkout_info['delivery_option'] = request.form['delivery-option']
        checkout_info['total_payment'] = request.form['total-payment']
        # Depending on the delivery option, get additional data
        if checkout_info['delivery_option'] == 'home-delivery':
            checkout_info['address'] = request.form['address']
            checkout_info['delivery_date'] = request.form['delivery-date']
            checkout_info['delivery_time'] = request.form['delivery-time']
        elif checkout_info['delivery_option'] == 'self-pickup':
            checkout_info['pickup_date'] = request.form['pickup-date']
            checkout_info['pickup_time'] = request.form['pickup-time']
        elif checkout_info['delivery_option'] == 'send-gift':
            checkout_info['sender_name'] = request.form['sender-name']
            checkout_info['sender_email'] = request.form['sender-email']
            checkout_info['sender_phone'] = request.form['sender-phone']
            checkout_info['receiver_name'] = request.form['receiver-name']
            checkout_info['receiver_email'] = request.form['receiver-email']
            checkout_info['receiver_phone'] = request.form['receiver-phone']
            checkout_info['receiver_address'] = request.form['receiver-address']

        session[cart_session_key] = checkout_info
        
        return redirect(url_for('confirm_order', website_id= website_id, cart_name=cart_name))

    return render_template('guest-checkout.html', website_id= website_id, cart_name = cart_name)

    
@app.route('/checkout-items/<website_id>/<cart_name>', methods=['GET', 'POST'])
def guest_checkout_items(website_id, cart_name):
    cart_session_key = f'cart_{website_id}_{cart_name}'
    checkout_session_key = f'checkout_{website_id}_{cart_name}'
    
    if request.method == 'POST':
        shipping_info = session.get(cart_session_key)
        product_info = request.get_json()
        
        # Generate and send OTP
        otp = str(random.randint(1000, 9999))
        subject = 'CakeMarque - Email Verification'
        message = f'Your OTP is: {otp}'
        mail.send_message(subject=subject, recipients=[shipping_info['email']], body=message)

        # Store OTP in session for verification
        session[checkout_session_key] = otp

        # Here, trigger your frontend to show an OTP input popup
        # This part depends on your frontend implementation

        return jsonify({'message': 'OTP sent. Please check your email.'}), 200
    
    elif request.method == 'GET':
        user_provided_otp = request.args.get('otp')
        if user_provided_otp and session.get(checkout_session_key) == user_provided_otp:
            # OTP is correct, create user and order objects
            user_data = {
                'name': shipping_info['name'],
                'phone': shipping_info['phone'],
                'email': shipping_info['email'],
                'register_status': 'unregistered',  
                'verification_status': 'unverified'
            }
            user_collection = db.users
            # Check if user already exists
            existing_user = user_collection.find_one({'email': user_data['email']})
            if not existing_user:
                # Insert new user if doesn't exist
                user_id = user_collection.insert_one(user_data).inserted_id
            else:
                # Use existing user's ID
                user_id = existing_user['_id']

            # Now create the order object
            order_data = {
                'name': product_info['name'],
                'image': product_info['image'],
                'quantity': product_info['quantity'],
                'size': product_info['size'],
                'flavor': product_info['flavor'],
                'caption': product_info['caption'],
                'node': product_info['note'],
                'user_id': ObjectId(user_id)  # Reference to user document
            }
            order_collection = db.orders
            order_collection.insert_one(order_data)

            return jsonify({'message': 'Order processed successfully'}), 200
        else:
            return jsonify({'message': 'Invalid OTP'}), 400
    else:
        return jsonify({'message': 'Invalid request method'}), 400



@app.route('/confirm-order/<website_id>/<cart_name>', methods=['GET'])
def confirm_order(website_id, cart_name):
    # Assuming you have a function to calculate the total payment
    total_payment = calculate_total_payment(f'cart_{website_id}_{cart_name}')
    half_payment = int(total_payment)/2
    cart_session_key = f'cart_{website_id}_{cart_name}'
    checkOutData = session[cart_session_key]
    print(checkOutData)
    if checkOutData['delivery_option'] == 'home-delivery' or checkOutData['delivery_option'] == 'self-pickup':
        message = f"Your Order has been Placed.At least 50% ({half_payment}) Advance Payment is required For Order Clearance. " \
                  "Below are the Account Details. Make Payment and send screenshot at WhatsApp or upload in the given field."
    else:
        message = f"Your Order has been Placed. 100% ({total_payment}) Advance Payment is required for sending Gift. " \
                  "Below are the Account Details. Make Payment and send screenshot at WhatsApp or upload in the given field."
    
    return render_template('confirm-order.html', total_payment= total_payment)
    
def calculate_total_payment(cart_session_key):
    # Retrieve cart data from the session
    # cart_data = session.get(cart_session_key, {})
    # print("Cart Data: ", cart_data)
    
    # total_payment = cart_data['total_payment']
    # print("total-payment: ", total_payment)
    total_payment = 100

    return total_payment


#paymob
import requests
from dotenv import load_dotenv
load_dotenv()
import requests

# Replace with your Paymob API credentials
api_key = os.environ.get('PAYMOB_API_KEY')
authentication_token = None  # Will be obtained during authentication

import uuid
import datetime

def generate_unique_order_id():
    """Generates a unique order ID using a combination of UUID and timestamp."""
    unique_id = str(uuid.uuid4())  # Generate a random UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]  # Formatted timestamp
    return f"{unique_id}_{timestamp}"  # Concatenate for a unique ID


@app.route('/initiate-payment', methods=['POST'])
def initiate_payment():
    try:
        data = request.json
        amount = data['amount']

        # Get authentication token if not already obtained
        global authentication_token
        if not authentication_token:
            response = requests.post('https://pakistan.paymob.com/api/auth/tokens', json={
                'api_key': api_key
            })
            
            response.raise_for_status()
            print("authentication response: ", response.text)
            authentication_token = response.json()['token']
            print("authentication token: ", authentication_token)
        
        

        # Create order
        headers = {'Authorization': 'Bearer ' + authentication_token}
        print("Headers: ", headers)

        response = requests.post('https://pakistan.paymob.com/api/ecommerce/orders', headers=headers, json={
            'amount_cents': amount * 100,
            'currency': 'PKR',  # Adjust currency code if needed
            'merchant_order_id': generate_unique_order_id()  # Implement your order ID generation logic
        })
        print("order response: ", response.text)
        
        response.raise_for_status()
        order_id = response.json()['id']
        # print("Order Id: ", order_id)

        # Get payment key
        response = requests.post('https://pakistan.paymob.com/api/acceptance/payment_keys', headers=headers, json={
            'order_id': order_id,
            'amount_cents': amount*100,  # Replace with the actual amount in cents
            "expiration": 3600,
            'currency': 'PKR',  # Adjust currency code if needed
            "billing_data": {
            "apartment": "29",
            "floor": "2", 
            "email": "yk221085@gmail.com",  
            "first_name": "Yasir", 
            "street": "Faisal Park", 
            "building": "8028", 
            "phone_number": "+923228043795", 
            "postal_code": "54950", 
            "city": "Lahore", 
            "country": "Pakistan", 
            "last_name": "Bahadar", 
            "state": "Punjab"}, 
            "integration_id": 141756

        })
        print("Payment key response: ", response.text)
        response.raise_for_status()
        payment_key = response.json()['token']
        print("payment key: ", payment_key)

        # Construct payment URL
        payment_url = f" https://pakistan.paymob.com/api/acceptance/iframes/153393?payment_token={payment_key}"

        return jsonify({'payment_url': payment_url})

    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Failed to initiate payment: ' + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)





if __name__ == '__main__':
    app.run(debug=True)
