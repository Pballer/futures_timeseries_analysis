import pandas as pd
import threading
from queue import Queue
import concurrent.futures
import random
import logging

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")


def producer(queue, url):
    results = random.randint(1, 100) # get_web_data(url) Call your function to load web data.
    logging.debug('putting %i' % results)
    queue.put(results)

def consumer(queue, event, database):
    while not event.is_set() or queue.qsize() != 0:
        try:
            results = queue.get(timeout=2)
            logging.info('Consumer: %i queue size: %i' % (results, queue.qsize()))
            # Call your function to save web data.
            # Save to sqlite/to_csv
        except Exception as e:
            logging.debug(e)
    logging.info('consumer exiting')


if __name__ == '__main__':
    queue = Queue(5)
    event = threading.Event()

    x = threading.Thread(target=consumer, args=(queue, event, 'database',))
    x.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        base_url = 'https://www.ramins_website.com/{yymmdd}.php'
        for date in pd.date_range(pd.datetime(2019, 8, 4), pd.datetime.today()):
            url = base_url.format(yymmdd=date.strftime('%y%m%d'))
            # Start producer thread to fetch one url.
            executor.submit(producer, queue, url)

    # Notify consumer that all producers have completed.
    event.set()
