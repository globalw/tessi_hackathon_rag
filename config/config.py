from dotenv import load_dotenv
import os

print("load .env" if load_dotenv(
    dotenv_path = "./.env"
) else "no .env found")

class Settings:
    DEBUG: bool = False

    def __init__(self, DEBUG: bool):
        self.DEBUG = DEBUG

settings = Settings(
    DEBUG = os.getenv("DEBUG", "false") == "true"
)

print(f"populated settings: {settings.__dict__}")