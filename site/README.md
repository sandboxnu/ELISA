instructions:
activate your virtualenv (normally source /env-name/bin/activate or similar)
pip install -r requirements.txt -- ensure you have all the requirements
python main.py

note: if something doesn't look right, you probably want python3 over python2 -- try using python3 main.py and pip3 install -r requirements.txt. it's entirely possible that the anaconda virtualenv has easy access to these dependencies through a GUI as well
