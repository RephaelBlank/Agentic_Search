import os
import logging

# Configure basic logging to provide feedback on key loading status
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Determine the execution environment dynamically
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def load_environment_keys():
    """
    Loads necessary API keys and tokens into the operating system's environment variables.
    
    If running within Google Colab, it securely fetches the keys using the Colab Secrets API.
    If running locally, it attempts to load them from a local .env file.
    """
    
    # The keys specified in the project's .env.example
    required_keys = [
        'TAVILY_API_KEY',
        'HF_TOKEN',
        'OPENAI_API_KEY',
        'GEMINI_API_KEY',
        'TOGETHER_API_KEY'
    ]

    if IN_COLAB:
        logging.info("Colab environment detected. Fetching API keys from Colab Secrets.")
        from google.colab import userdata
        
        for key in required_keys:
            try:
                secret_value = userdata.get(key)
                if secret_value:
                    os.environ[key] = secret_value
                else:
                    logging.warning(f"Secret '{key}' was found but contains an empty value.")
            except userdata.SecretNotFoundError:
                logging.warning(f"Secret '{key}' is missing from Colab Secrets. Please add it.")
                
    else:
        logging.info("Local environment detected. Attempting to load from .env file.")
        try:
            from dotenv import load_dotenv
            
            # Load variables from .env into os.environ
            load_dotenv() 
            
            # Verify which keys were successfully loaded
            for key in required_keys:
                if not os.getenv(key):
                    logging.warning(f"Key '{key}' is missing from the local .env file.")
                    
        except ImportError:
            logging.error("The 'python-dotenv' package is missing. Please install it using 'pip install python-dotenv' for local development.")

# Automatically execute the loading process when this module is imported
load_environment_keys()