from flask_mongoengine import MongoEngine
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
db = MongoEngine()

from datetime import datetime, timedelta
import secrets


# serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
class User(db.Document):
    username = db.StringField(unique=True, required=True)
    email = db.StringField(unique=True, required=True)
    password = db.StringField(required=True)
    role = db.StringField(required=True, default="customer")  # "customer" or "admin"
    status = db.StringField(required=True, default="unverified")
    # ... (rest of the class) ...
    
    def set_password(self, password):
        self.password = generate_password_hash(password, method='sha256')
    
    def check_password(self, password):
        return check_password_hash(self.password, password)


class Profile(db.Document):
    user = db.ReferenceField(User, unique=True)
    name = db.StringField(required=True)
    phone_number = db.StringField()
    address = db.StringField()
    # Other profile fields

class Cart(db.Document):
    user = db.ReferenceField(User)
    product = db.ReferenceField('Product')
    quantity = db.IntField(required=True)
    flavor = db.StringField()
    cake_size = db.StringField()
    message = db.StringField()
    total_price = db.FloatField()

from mongoengine import Document, StringField, FloatField, ListField


class Product(db.Document):
    product_id = db.StringField()
    product_image = db.StringField()  # File path or URL to cake image
    product_name = db.StringField(required=True)
    categories = db.ListField(StringField())
    constant_price = db.FloatField(required=True)
    short_description = db.StringField()
    long_description = db.StringField()
    review_images = db.ListField(StringField())
    minimum_size = db.IntField(required=True)

class Counter(db.Document):
    name = db.StringField(unique=True)
    value = db.IntField(default=0)

class Flavors(db.Document):
    name = db.StringField(required=True, unique=True)
    price = db.FloatField(required=True)

class Shipping(db.Document):
    area = db.StringField(required=True, unique=True)
    charges = db.FloatField(required=True)

class Order(db.Document):
    user = db.ReferenceField(User)
    product = db.ReferenceField(Product)
    quantity = db.IntField(required=True)
    flavor = db.StringField()
    cake_size = db.StringField()
    message = db.StringField()
    total_price = db.FloatField()
    shipping_address = db.StringField()
    order_status = db.StringField(default="pending")  # "pending", "paid", "shipped", etc.

    # Other order fields (e.g., payment info, timestamps)
