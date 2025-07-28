sudo gunicorn cnn_ctc:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
