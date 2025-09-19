from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

    # Uploads folder
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    # CORS
    CORS(app, resources={r"/*": {"origins": "http://localhost:4000"}})

    # Register routes
    from routes.predict import predict_bp
    from routes.predict_video import predict_video_bp
    app.register_blueprint(predict_bp)
    app.register_blueprint(predict_video_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
