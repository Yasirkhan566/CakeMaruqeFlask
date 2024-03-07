from flask import render_template, request, redirect, url_for, flash, Flask, jsonify, session
from models import User, Profile, Cart, Product, Order, Counter, Shipping, Flavors
from flask_restful import Resource, reqparse
from models import Product
import json
import os

from werkzeug.utils import secure_filename

class GetLatestProductsByCategory(Resource):
    def get(self, category):
        # Fetch the 10 latest products for the given category
        latest_products = Product.objects(categories=category).order_by('-created_at').limit(10)

        formatted_products = []
        for product in latest_products:
            formatted_products.append({
                'id': product.product_id,
                'image': product.product_image,
                'name': product.product_name,
                'price': calculate_price(product),  # Implement your price calculation logic
                'rating': get_product_rating(product),  # Implement this function later
            })

        return jsonify({'latest_products': formatted_products})
def calculate_price(product):
    # Calculate the price based on your formula
    # Price = constant_price + (minimum_price_per_pound * minimum_size)
    # You need to fetch the minimum_price_per_pound and minimum_size from the respective collections
    # Implement this function according to your database structure
    return product['constant_price'] + (1600 * (product['minimum_size']))

def get_product_rating(product_id):
    # Fetch the product rating from the ratings collection based on the product_id
    # Implement this function later
    return 4.5  # Dummy value for demonstration


from flask import current_app
from itsdangerous import URLSafeTimedSerializer


class UserRegisterResource(Resource):
    def post(self):
        data = request.get_json()

        if User.objects(username=data['username']).first() or User.objects(email=data['email']).first():
            return {'message': 'Username or Email Already Registered', 'status': 'error'}, 400

        # Create a user
        user = User(username=data['username'], email=data['email'])
        user.set_password(data['password'])
        user.save()

        # Generate verification token
        serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
        verification_token = generate_verification_token(data['email'], serializer)

        # Send verification email using the new resource endpoint
        verification_link = url_for('emailverificationresource', verification_token=verification_token, _external=True)
        subject = 'CakeMarque - Email Verification'
        message = f'Click the following link to verify your email: {verification_link}'
        from app import mail
        mail.send_message(subject=subject, recipients=[data['email']], body=message)

        return {'message': 'Registration successful. Please check your email for verification link.', 'status': 'success'}, 200


class UserLoginResource(Resource):
    def post(self):
        data = request.get_json()
        user = User.objects(username=data['username']).first() or User.objects(email=data['username']).first()
        if user and user.check_password(data['password']) and user.status == 'verified':
            session['user_id'] = str(user.id)
            if str(user.role) == 'customer':
                return {'message': 'Login successful', 'redirect': url_for('customer_dashboard')}, 200
            elif str(user.role) == 'admin':
                return {'message': 'Login successful', 'redirect': url_for('admin_dashboard')}, 200
        elif user and user.check_password(data['password']) and user.status == 'unverified':
            return {'message': 'Unable to Login. Please verify your email first...', 'redirect': url_for('verify_email')}, 200
        return {'message': 'Invalid credentials','redirect': url_for('login')}, 200


# Function to generate a verification token
def generate_verification_token(email, serializer):
    return serializer.dumps(email, salt='email-verify')




class EmailVerificationResource(Resource):
    def __init__(self, app):
        self.app = app
    def get(self, verification_token):
        serializer = URLSafeTimedSerializer(self.app.config['SECRET_KEY'])

        try:
            email = serializer.loads(verification_token, salt='email-verify', max_age=3600)
            user = User.objects(email=email).first()
            if user:
                user.status = "verified"
                user.save()
                flash('Email verification successful. You can now log in.', 'success')
            else:
                flash('Verification failed.', 'error')  # Use the appropriate flash function

        except BadSignature:
            flash('Verification failed.', 'error')  # Use the appropriate flash function
        
        return redirect(url_for('login'))
class SendVerificationLink(Resource):
    def post(self):
        data = request.get_json()

        if User.objects(email=data['email']).first():
            # Generate verification token
            serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
            verification_token = generate_verification_token(data['email'], serializer)

            # Send verification email using the new resource endpoint
            verification_link = url_for('emailverificationresource', verification_token=verification_token, _external=True)
            subject = 'CakeMarque - Email Verification'
            message = f'Click the following link to verify your email: {verification_link}'
            from app import mail
            mail.send_message(subject=subject, recipients=[data['email']], body=message)

            return {'message': 'Verification Link sent at your email address', 'redirect': url_for('verify_email')}, 200
        else:
            return {'message': 'Unable to find email address. Please register first...', 'redirect': url_for('register')}, 400
            
class CustomerDashboardResource(Resource):
    def get(self):
        user_id = session.get('user_id')
        if user_id:
            user = User.objects(id=user_id).first()
            return f'Welcome, {user.username}!'

# class AdminDashboardResource(Resource):
#     def get(self):
#         user_id = session.get('user_id')
#         if user_id:
#             user = User.objects(id=user_id).first()
#             return f'Welcome, {user.username}!'




class InsertProductResource(Resource):
    def get(self, form_id):
        return render_template('insert_product.html')

    def post(self):
        counter = Counter.objects(name="product_counter").first()

        if counter is None:
            # Create the counter if it doesn't exist
            counter = Counter(name="product_counter", value=0)
            counter.save()

        # Increment the counter value
        counter.value += 1
        counter.save()

# Now, counter.value contains the updated value


        # Generate the product ID
        product_id = f'CML{counter.value}'
        product_image = request.form.get('product_selected_image_paths')
        print("Product Image: ", product_image)
        product_name = request.form.get('product_name')
        constant_price = float(request.form.get('constant_price'))
        short_description = request.form.get('short_description')
        long_description = request.form.get('long_description')
        categories = request.form.getlist('categories')
        review_images = request.form.get('selected_image_paths')
        review_images = review_images.split(',')
        minimum_size = request.form.get('minimum_size')
        print(review_images)
        from app import app
        print("Product Image: ", product_image)
        print("review Images", review_images)

        
        product = Product(
            product_id = product_id,
            product_image=product_image,
            product_name=product_name,
            constant_price=constant_price,
            categories=categories,
            short_description = short_description,
            long_description = long_description,
            review_images = review_images,
            minimum_size = minimum_size
        
        )
        product.save()
        
        return {'message': 'Product added successfully.', 'status': 'success'}, 200
    
class InsertDCResource(Resource):
    def get(self, form_id):
        return render_template('insert_product.html')

    def post(self):
        flavor_name = request.form.get('flavor_name')
        flavor_charges = float(request.form.get('flavor_charges'))
        
        shipping = Shipping(
            name=flavor_name,
            charges=flavor_charges,
        
        )
        shipping.save()
        
        return {'message': 'Shipping Entry added successfully.', 'status': 'success'}, 200
    
class InsertFlavorResource(Resource):
    def post(self):
        flavor_name = request.form.get('flavor-name')
        flavor_price = float(request.form.get('flavor-price'))
        flavor = Flavors(
            name = flavor_name,
            price = flavor_price,
        )
        flavor.save()
        return {'message': 'Flavor added successfully.', 'status,': 'success'}, 200
    
class InsertShippingResource(Resource):
    def post(self):
        shipping_area = request.form.get('shipping-area')
        shipping_charges= float(request.form.get('shipping-charges'))
        shipping = Shipping(
            area = shipping_area,
            charges = shipping_charges,
        )
        shipping.save()
        return {'message': 'Area added successfully.', 'status,': 'success'}, 200

class SubmitAllProductsResource(Resource):
    def post(self):
        product_data = request.form.get('product_data')
        print(product_data)
        products = json.loads(product_data)
        
        for product in products:
            # Handle image fields and review_images list
            product_image = product.pop('product_image', None)
            review_images = product.pop('review_images', [])

            new_product = Product(**product)

            if product_image:
                new_product.product_image = product_image

            if review_images:
                new_product.review_images = [str(image) for image in review_images]  # Convert image paths to strings
            
            new_product.save()
        
        return {'message': 'All products added successfully.', 'status': 'success'}, 200





# Resource to fetch product data
class ProductListResource(Resource):
    def get(self):
        products = Product.objects.all()
        product_data = []
        for product in products:
            product_data.append({
                'product_image': product.product_image,
                'product_name': product.product_name,
                'short_description': product.short_description,
                'product_id': product.product_id,
                'categories': product.categories,
                'constant_price': product.constant_price,
            })
        return jsonify(product_data)

class ProductDeleteResource(Resource):
    def delete(self, product_id):
        print("Product ID to Delete",product_id)
        try:
            # Find the product by ID
            product = Product.objects.get(product_id=product_id)

            # Delete the product
            product.delete()

            return {'message': 'Product deleted successfully'}, 200
        

        except Product.DoesNotExist:
            return {'error': 'Product not found'}, 404

        except Exception as e:
            return {'error': 'An error occurred while deleting the product', 'details': str(e)}, 500
        
class ProductEditResource(Resource):
    def get(self, product_id):
        try:
            # Find the product by ID
            product = Product.objects.get(product_id=product_id)
            # Return product details as JSON
            return {
                'product_id': str(product.product_id),
                'product_image': product.product_image,
                'product_name': product.product_name,
                'constant_price': product.constant_price,
                'short_description': product.short_description,
                'long_description': product.long_description,
                'minimum_size': product.minimum_size,
                
            }, 200
            

        except Product.DoesNotExist:
            return {'error': 'Product not found'}, 404

        except Exception as e:
            return {'error': 'An error occurred while fetching product details', 'details': str(e)}, 500

    def put(self, product_id):
        args = self.parser.parse_args()

        try:
            # Find the product by ID
            product = Product.objects.get(product_id=product_id)

            # Update the product fields
            product.product_name = args['product_name']
            product.constant_price = args['constant_price']
            # Update other fields as needed

            # Save the changes
            product.save()

            return {'message': 'Product updated successfully'}, 200

        except Product.DoesNotExist:
            return {'error': 'Product not found'}, 404

        except Exception as e:
            return {'error': 'An error occurred while updating the product', 'details': str(e)}, 500



class GetExistingImagesResource(Resource):
    def get(self):
        # Directory where your images are stored
        images_directory = 'D:/downloads/desktop 2/cakemarque flask mongodb/cakemarque flask mongodb/static/images'

        # Get a list of image file names in the directory
        image_files = [filename for filename in os.listdir(images_directory) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        # Create full image paths using the directory path and file names
        image_paths = [('static/images/'+ filename) for filename in image_files]
        return image_paths


def get_all_image_filenames():
    image_folder = 'D:/downloads/desktop 2/cakemarque flask mongodb/cakemarque flask mongodb/static/images'  # Replace this with the actual path to your image folder
    image_filenames = []

    for filename in os.listdir(image_folder):
        print("File Names:")
        print(filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_filenames.append("images/"+filename)

    return image_filenames

def generate_updated_gallery_html(image_filenames):
    return render_template('gallery-viewer.html', images=image_filenames)

class UploadImagesResource(Resource):
    def post(self):
        from app import app
        app.config['UPLOAD_FOLDER'] = 'D:/downloads/desktop 2/cakemarque flask mongodb/cakemarque flask mongodb/static/images'
        for image in request.files.getlist('upload_images'):
            if image:
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # # Generate and return the updated HTML content for the gallery
        # image_filenames = get_all_image_filenames()
        # updated_gallery_html = generate_updated_gallery_html(image_filenames)
        # return jsonify(html=updated_gallery_html)





        