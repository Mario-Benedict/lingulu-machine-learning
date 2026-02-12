"""
Authentication middleware for verifying user authentication via Spring Boot service.
"""
from functools import wraps
from flask import request, jsonify
import httpx
from typing import Optional, Callable

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AuthenticationMiddleware:
    """
    Middleware to verify authentication by making requests to Spring Boot service.
    """
    
    def __init__(self, auth_service_url: str, timeout: float = 5.0):
        """
        Initialize authentication middleware.
        
        Args:
            auth_service_url: URL of Spring Boot authentication service
            timeout: Request timeout in seconds
        """
        self.auth_service_url = auth_service_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"Authentication middleware initialized with service URL: {self.auth_service_url}")
    
    def verify_cookie(self, cookie_value: str) -> tuple[bool, Optional[dict]]:
        """
        Verify authentication cookie with Spring Boot service.
        
        Args:
            cookie_value: Cookie value to verify
            
        Returns:
            Tuple of (is_authenticated, user_data)
        """
        try:
            # Make request to Spring Boot service for verification
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.auth_service_url}/api/account/authenticated",
                    cookies={"token": cookie_value},  # Adjust cookie name as needed
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    logger.info(f"Authentication successful for user: {user_data.get('userId', 'unknown')}")
                    return True, user_data
                elif response.status_code == 401:
                    logger.warning("Authentication failed: Unauthorized")
                    return False, None
                else:
                    logger.error(f"Unexpected response from auth service: {response.status_code}")
                    return False, None
                    
        except httpx.TimeoutException:
            logger.error(f"Authentication service timeout after {self.timeout}s")
            return False, None
        except httpx.RequestError as e:
            logger.error(f"Error connecting to authentication service: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}")
            return False, None
    
    def require_auth(self, f: Callable) -> Callable:
        """
        Decorator to protect routes with authentication.
        
        Usage:
            @app.route('/protected')
            @auth_middleware.require_auth
            def protected_route():
                # Access user data from request.user_data
                return jsonify({'message': 'Success'})
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get cookie from request (adjust cookie name as needed)
            cookie_value = request.cookies.get('token')
            
            if not cookie_value:
                logger.warning("Authentication failed: No cookie found")
                return jsonify({
                    'error': 'Authentication required',
                    'message': 'No authentication cookie found'
                }), 401
            
            # Verify with Spring Boot service
            is_authenticated, user_data = self.verify_cookie(cookie_value)
            
            if not is_authenticated:
                return jsonify({
                    'error': 'Authentication failed',
                    'message': 'Invalid or expired authentication'
                }), 401
            
            # Store user data in request context for use in route
            request.user_data = user_data
            
            return f(*args, **kwargs)
        
        return decorated_function


def setup_auth_middleware(auth_service_url: str, timeout: float = 5.0) -> AuthenticationMiddleware:
    """
    Factory function to create and configure authentication middleware.
    
    Args:
        auth_service_url: URL of Spring Boot authentication service
        timeout: Request timeout in seconds
        
    Returns:
        Configured AuthenticationMiddleware instance
    """
    return AuthenticationMiddleware(auth_service_url, timeout)
