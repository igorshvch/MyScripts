def worker(lock, message=None):
    with lock:
        print(message)