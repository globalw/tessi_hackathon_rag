from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    DEBUG: bool = False

settings = Settings()

if(os.getenv("DEBUG") == "true"):
    settings.DEBUG = True