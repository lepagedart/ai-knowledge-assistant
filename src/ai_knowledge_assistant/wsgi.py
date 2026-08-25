"""Production WSGI entry point; importing it never calls OpenAI."""

from .web import create_app

app = create_app()
