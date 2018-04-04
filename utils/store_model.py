import time
import utils.store_file as sf

def save_metrics_to_file(filePath, description, id, metrics):
    metrics.insert(0, description)
    metrics.insert(0, id)
    metrics.insert(0, int(time.time()))
    sf.write_to_file(metrics,filePath)