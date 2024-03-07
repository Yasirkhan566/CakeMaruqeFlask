from resources import (
    UserRegisterResource,
    UserLoginResource,
    CustomerDashboardResource,
    # AdminDashboardResource,
    EmailVerificationResource,
    SendVerificationLink,
    InsertProductResource,
    SubmitAllProductsResource,
    ProductListResource,
    UploadImagesResource,
    GetExistingImagesResource,
    GetLatestProductsByCategory,
    ProductDeleteResource,
    ProductEditResource,
    InsertDCResource,
    InsertFlavorResource,
    InsertShippingResource,

)

def initialize_routes(api):
    # Add resources to the API
    api.add_resource(UserRegisterResource, '/api/register')
    api.add_resource(UserLoginResource, '/api/login')
    api.add_resource(CustomerDashboardResource, '/api/customer-dashboard')
    # api.add_resource(AdminDashboardResource, '/api/admin-dashboard')
    from app import app
    api.add_resource(EmailVerificationResource, '/api/verify-email/<string:verification_token>', resource_class_kwargs={'app': app})
    api.add_resource(SendVerificationLink, '/api/verify-email')
    api.add_resource(InsertProductResource, '/api/insert-product')
    api.add_resource(InsertFlavorResource, '/api/insert-flavor')
    api.add_resource(InsertShippingResource, '/api/insert-shipping')
    api.add_resource(InsertDCResource, '/api/insert-delivery-charges')
    api.add_resource(SubmitAllProductsResource, '/api/submit-all-products')
    api.add_resource(ProductListResource, '/api/products')
    api.add_resource(UploadImagesResource, '/api/upload-images')
    api.add_resource(GetExistingImagesResource, '/api/get-existing-images')
    api.add_resource(GetLatestProductsByCategory, '/api/get_latest_products/<category>')
    api.add_resource(ProductDeleteResource, '/api/products/<product_id>')
    api.add_resource(ProductEditResource, '/api/products/<product_id>')