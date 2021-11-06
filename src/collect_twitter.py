import datetime
import requests
import os
import time
import pandas as pd
import json

# Replace with Bearer Token or use environment variable
bearer_token = ""

CSV_FILE = "tweets_states_weekbefore.csv"
TIMESTAMP_FILE = "collection_timestamps_oct2020.txt"
METADATA_FILE = "meta_semeval.json"

search_url = "https://api.twitter.com/2/tweets/search/all"

query_string = 'Trump lang:en -is:retweet place:"{}"'
query_fields = "id,author_id,text,conversation_id,created_at,referenced_tweets,geo"
time_window = 12


def construct_query(place, query_starttime, query_endtime):
    # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
    # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
    query_params = {
        "query": query_string.format(place),
        "tweet.fields": query_fields,
        "start_time": query_starttime.isoformat(),
        "end_time": query_endtime.isoformat(),
        "max_results": 100,
    }
    return query_params


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", search_url, headers=headers, params=params)
    print(response.status_code)
    if response.status_code != 200:
        print(response.status_code, response.text)
        return None
    return response.json()


def batch_of_requests(place):
    headers = create_headers(bearer_token)
    requests_used = 0

    with open(TIMESTAMP_FILE) as f:
        timestamps = f.read().split()
        if not timestamps:
            return requests_used
        timestamps = [datetime.datetime.fromisoformat(x) for x in timestamps]
    for _ in range(len(timestamps)):
        time.sleep(1.0)
        if len(timestamps) == 0:
            break
        timestamp = timestamps.pop(0)
        query_params = construct_query(
            place, timestamp, timestamp + datetime.timedelta(hours=time_window)
        )
        requests_used += 1
        json_response = connect_to_endpoint(search_url, headers, query_params)
        if not json_response:
            print("No Json Response")
            continue
        if "data" not in json_response:
            print(json_response)
            continue
        # print(json.dumps(json_response, indent=4, sort_keys=True))
        df = pd.DataFrame.from_records(json_response["data"])
        df["Place"] = place
        if os.path.exists(CSV_FILE):
            df_big = pd.read_csv(CSV_FILE, lineterminator="\n")
            df_big = df_big.append(df)
            df_big.to_csv(CSV_FILE, index=False)
        else:
            df.to_csv(CSV_FILE, index=False)
        with open(METADATA_FILE, "w") as f:
            json.dump(json_response["meta"], f)
        print("Num of records retrieved:", len(df))
    with open(TIMESTAMP_FILE, "w") as f:
        f.write("\n".join([x.isoformat() for x in timestamps]))
    return requests_used


def hours():
    d = datetime.datetime.fromisoformat("2020-10-27T00:00:00+00:00")
    enddate = datetime.datetime.fromisoformat("2020-11-03T00:00:00+00:00")
    delta = datetime.timedelta(hours=time_window)
    f = open(TIMESTAMP_FILE, "w")
    while d < enddate:
        print(d.isoformat(), file=f)
        d += delta
    f.close()


if __name__ == "__main__":

    with open("states.txt") as f:
        states = f.read().strip().split("\n")[17:]

    requests_used = 0
    for state in states:
        hours()
        requests_used += batch_of_requests(state)
        # must be spaced out by 15 mins
        if requests_used > 280:
            time.sleep(900)
            requests_used = 0
