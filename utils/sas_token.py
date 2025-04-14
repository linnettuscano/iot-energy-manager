import os
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
from utils.logger import setup_logger

logger = setup_logger("SASToken")
TOKEN_FILE = "azure_sas_token.json"

def generate_sas_token(uri, key, expiry_in_seconds=3600):
    ttl = int(time.time() + expiry_in_seconds)
    sign_key = f"{urllib.parse.quote_plus(uri)}\n{ttl}".encode("utf-8")
    decoded_key = base64.b64decode(key)
    signature = base64.b64encode(hmac.new(decoded_key, sign_key, hashlib.sha256).digest())
    signature = urllib.parse.quote_plus(signature)
    token = f"SharedAccessSignature sr={urllib.parse.quote_plus(uri)}&sig={signature}&se={ttl}"
    return token, ttl

def get_stored_token():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                return data.get("sas_token"), data.get("se")
        except Exception as e:
            logger.error("Error reading SAS token file: %s", e)
    return None, None

def store_token(token, se):
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({"sas_token": token, "se": se}, f)
    except Exception as e:
        logger.error("Error storing SAS token to file: %s", e)

def get_sas_token(host, device_id, primary_key, validity_period=3600):
    """
    Check if a valid SAS token is stored.
    If valid token exists (with a buffer of 60 seconds), return it.
    Otherwise, generate a new token, store it, and return it.
    """
    resource_uri = f"{host}/devices/{device_id}"
    current_time = int(time.time())
    
    stored_token, expiry = get_stored_token()
    if stored_token and expiry:
        if current_time < int(expiry) - 60:
            logger.info("Using stored SAS token, expires at %d", int(expiry))
            return stored_token
        else:
            logger.info("Stored SAS token expired or nearly expired. Generating a new one.")
    else:
        logger.info("No stored SAS token found. Generating a new one.")
    
    new_token, new_expiry = generate_sas_token(resource_uri, primary_key, validity_period)
    store_token(new_token, new_expiry)
    logger.info("New SAS token generated, expires at %d", new_expiry)
    return new_token