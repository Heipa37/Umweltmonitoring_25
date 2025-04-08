Projekt setup:

-Prequesits:
>>> pip install virtualenv

-Virtual environement:
create: python3.xx -m venv my_venv_name
>>> python3.12 -m venv venv_UM25
activate: source my_venv_name/bin/activate
>>> source venv_UM25/bin/activate

- Modules
to load from a existing requirements file:
>>> pip install -r requirements.txt
to crate a new requirements file:
>>> pip freeze > requirements.txt

-Gitignore
create Gitignore:
>>> touch .gitignore
comments in gitignore start with #
list all files, you don't want to push