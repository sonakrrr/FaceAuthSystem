import hashlib
import hmac
import os

class SecurityManager:

    SEPARATOR = "$"
    ENCODING  = "utf-8"
    SALT_SIZE = 16

    def hash_password(self, password):

        salt = os.urandom(self.SALT_SIZE)
        salted = salt + password.encode(self.ENCODING)
        hash_bytes = hashlib.sha256(salted).digest()

        return f"{salt.hex()}{self.SEPARATOR}{hash_bytes.hex()}"

    def verify_password(self, password, stored_hash):

        try:
            parts = stored_hash.split(self.SEPARATOR)
            if len(parts) != 2:
                return False

            salt_hex, hash_hex = parts
            salt = bytes.fromhex(salt_hex)

            salted = salt + password.encode(self.ENCODING)
            hash_bytes = hashlib.sha256(salted).digest()

            return hmac.compare_digest(hash_bytes.hex(), hash_hex)

        except (ValueError, AttributeError):
            return False

    def hash_username(self, username):

        return hashlib.sha256(
            username.encode(self.ENCODING)
        ).hexdigest()


if __name__ == "__main__":
    security = SecurityManager()

    print("=== Cryptographic Security Module Test ===\n")

    password = "my_secret_password"
    stored   = security.hash_password(password)
    print(f"Source Password: {password}")
    print(f"Generated Hash:  {stored}\n")

    result = security.verify_password(password, stored)
    print(f"Valid Password Verification:   {result}")

    result_wrong = security.verify_password("wrong_password", stored)
    print(f"Invalid Password Verification: {result_wrong}\n")

    stored2 = security.hash_password(password)
    print(f"Hash Sample 1: {stored[:32]}...")
    print(f"Hash Sample 2: {stored2[:32]}...")
    print(f"Hashes are unique due to salt: {stored != stored2}")

    username      = "sofia"
    username_hash = security.hash_username(username)
    print(f"\nUsername string: {username}")
    print(f"Username Hash:   {username_hash}")
    print(f"Deterministic integrity verification: "
          f"{username_hash == security.hash_username(username)}")