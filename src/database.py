from dataclasses import dataclass
from pymongo import MongoClient
import numpy as np
from datetime import datetime


# Used to store results in mongodb
def apply_marks(ls: list):
    if type(ls) == np.ndarray:
        return ["np.ndarray", ls.tolist()]
    elif type(ls) == list:
        return [apply_marks(l) for l in ls]
    else:
        return ls


# Used to convert back from mongodb stored results
def revert_marks(ls: list):
    if type(ls) == list and len(ls) == 2 and ls[0] == "np.ndarray" and type(ls[1]) == list:
        return np.array(ls[1])
    elif type(ls) == list:
        return [revert_marks(l) for l in ls]
    else:
        return ls


@dataclass
class FischerResult:
    '''Class to store a single fischer result.
    Use a list of this class to store many results.'''
    observable: np.ndarray
    times: np.ndarray
    parameters: list
    q_arr: list
    constants: list
    y0: np.array

    def to_savedict(self):
        '''Used to store results in database'''
        d = {
            "observable": apply_marks(self.observable),
            "times": apply_marks(self.times),
            "parameters": apply_marks(self.parameters),
            "q_arr": apply_marks(self.q_arr),
            "constants": apply_marks(self.constants),
            "y0": apply_marks(self.y0)
        }
        return d


def convert_fischer_results(fischer_results):
    '''Converts results stored in database to fischer_results.
    The conversion is NOT 1:1 since numpy arrays will not be arrays afterwards due to
    mongodb not being able to directly store numpy arrays.'''
    return [FischerResult(*f[0]) for f in fischer_results]


def __get_mongodb_client():
    client = MongoClient('pleyer-ws.fdm.privat', 27017)
    return client


def __get_mongodb_database():
    client = __get_mongodb_client()
    # This should probably for the future be modified to have custom names
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
    fisses = [f.to_savedict() for f in fischer_dataclasses]
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


def list_all_collections():
    db = __get_mongodb_database()
    print(db.list_collection_names())


def get_fischer_results_from_collection(collection):
    coll = get_collection(collection)
    fisses = [[[revert_marks(c[key]) for key in ["observable", "times", "parameters", "q_arr", "constants", "y0"]]] for c in coll.find()]
    return fisses
