import pickle
from pathlib import path

import streamlit_authenticator as stauth

names = ["tsani", "dimas"]
username = ["ttsan","ddimas"]
password = ["abc123","abc234"]

hashed_passwords = stauth.Hasher(password).generate()

file_path = path(__file__).parent/"hashed_pw.pkl"
with file_path.open("wb")as file:
    pickle.dump(hashed_passwords, file)