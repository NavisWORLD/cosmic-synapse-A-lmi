"""
Security Layer: Encryption utilities
AES-256 for data at rest, TLS for network communication
"""

import logging
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Encryption manager for A-LMI system.
    
    Uses AES-256 for data at rest encryption.
    Keys are managed by KMS (HashiCorp Vault).
    """
    
    def __init__(self, kms_provider=None):
        """
        Initialize encryption manager.
        
        Args:
            kms_provider: Key management service provider (Vault client)
        """
        self.kms_provider = kms_provider
        self.cipher: Optional[Fernet] = None
        
    def load_key(self, key_name: str = "default"):
        """
        Load encryption key from KMS.
        
        Args:
            key_name: Key name in KMS
        """
        if self.kms_provider:
            # Retrieve key from Vault
            try:
                secret = self.kms_provider.secrets.kv.v2.read_secret_version(path=key_name)
                key = secret['data']['data']['key'].encode()
                
                self.cipher = Fernet(key)
                logger.info(f"Loaded encryption key: {key_name}")
            except Exception as e:
                logger.error(f"Error loading key from KMS: {e}")
                raise
        else:
            # Fallback: generate key from password
            # WARNING: This is not production-ready
            logger.warning("No KMS provider, using fallback key generation")
            password = b"a_lmi_default_password_change_in_production"
            salt = b"a_lmi_salt"
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.cipher = Fernet(key)
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Plaintext data
            
        Returns:
            Encrypted data
        """
        if not self.cipher:
            raise RuntimeError("Encryption key not loaded")
        
        try:
            encrypted_data = self.cipher.encrypt(data)
            return encrypted_data
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Plaintext data
        """
        if not self.cipher:
            raise RuntimeError("Encryption key not loaded")
        
        try:
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return decrypted_data
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def encrypt_string(self, text: str) -> str:
        """
        Encrypt string.
        
        Args:
            text: Plaintext string
            
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.encrypt(text.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt string.
        
        Args:
            encrypted_text: Base64-encoded encrypted string
            
        Returns:
            Plaintext string
        """
        encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
        decrypted = self.decrypt(encrypted_data)
        return decrypted.decode('utf-8')

