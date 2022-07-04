from dataclasses import dataclass
from pymongo import MongoClient
import numpy as np
from datetime import datetime


class FischerResult:
    '''Class to store a single fischer result.
    Use a list of this class to store many results.'''

    def __init__(self, observable: np.ndarray, times: np.ndarray, parameters: list, q_arr: list, constants: list, y0: np.ndarray):
        self.observable = observable.tolist()
        self.times = times.tolist()
        self.parameters = [p.tolist() if type(p) == np.ndarray else p for p in parameters]
        self.q_arr = [q.tolist() if type(q) == np.ndarray else q for q in q_arr]
        self.constants = [c.tolist() if type(c) == np.ndarray else c for c in constants]
        self.y0 = y0.tolist()

    def to_dict(self):
        d = {
            "observable": self.observable,
            "times": self.times,
            "parameters": self.parameters,
            "q_arr": self.q_arr,
            "constants": self.constants,
            "y0": self.y0
        }
        return d
    
    def to_list(self):
        return  [self.observable, self.times, self.parameters, self.q_arr, self.constants, self.y0]


def convert_fischer_results(fischer_results):
    fischer_dataclasses = []
    for f in fischer_results:
        fischer_dataclasses.append(FischerResult(*(f[0])))
    return fischer_dataclasses


def __get_mongodb_client():
    client = MongoClient('localhost', 27017)
    return client


def __get_mongodb_database():
    client = __get_mongodb_client()
    db = client.tsenso_pgaindrik_model_design
    return db


def generate_new_collection(name: str):
    if len(name) < 4:
        raise ValueError("Name too small. Choose a descriptive name with at least 4 characters")
    # Name collection with current time
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d-%H:%M:%S_")
    collname = dt_string + name
    # Store the collection in the database responsible named tsenso_pgaindrik_model_design
    db = __get_mongodb_database()
    collist = db.list_collection_names()
    if collname in collist:
        raise ValueError("The collection with the name " + collname + " already exists!")
    collection = db[collname]
    print("Created collection with name " + collname)
    return collection


def insert_fischer_dataclasses(fischer_dataclasses, collection):
    coll = get_collection(collection)
    fisses = [f.to_dict() for f in fischer_dataclasses]
    coll.insert_many(fisses)


def drop_all_collections():
    db = __get_mongodb_database()
    collist = db.list_collection_names()
    for name in collist:
        db.drop_collection(name)


def get_collection(collection):
    db = __get_mongodb_database()
    if type(collection) == str:
        if collection not in db.list_collection_names():
            print("Currently stored collections (names):")
            print(db.list_collection_names())
            raise ValueError("No collection with the name " + collection + " found.")
        else:
            return db[collection]
    else:
        return collection


def get_fischer_results_from_collection(collection):
    coll = get_collection(collection)
    fisses = [[[np.array(c[key]) for key in ["observable", "times", "parameters", "q_arr", "constants", "y0"]]] for c in coll.find()]
    return fisses