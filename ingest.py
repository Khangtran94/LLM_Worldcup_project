import requests
import json
import dlt
import os

# List of years in openfootball/worldcup.json repo
YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 
    1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 
    2018, 2022, 2025, 2026
]

def fetch_worldcup_data():
    all_matches = []
    all_goals = []
    
    for year in YEARS:
        # Try worldcup-full.json first, then worldcup.json
        urls = [
            f"https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup-full.json",
            f"https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"
        ]
        
        data = None
        for url in urls:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    print(f"Successfully fetched data for {year} from {url.split('/')[-1]}")
                    break
            except Exception as e:
                continue
                
        if not data:
            print(f"Warning: Could not fetch data for year {year}")
            continue
            
        tournament_name = data.get("name", f"World Cup {year}")
        matches_list = data.get("matches", [])
        
        for idx, match in enumerate(matches_list):
            team1 = match.get("team1")
            team2 = match.get("team2")
            if not team1 or not team2:
                continue
                
            # Create a unique match ID
            t1_slug = team1.lower().replace(" ", "_")
            t2_slug = team2.lower().replace(" ", "_")
            match_id = f"{year}_{t1_slug}_vs_{t2_slug}_{idx}"
            
            # Extract scores
            score = match.get("score")
            score_team1 = None
            score_team2 = None
            score_ht_team1 = None
            score_ht_team2 = None
            
            if isinstance(score, list):
                score_team1 = score[0] if len(score) > 0 else None
                score_team2 = score[1] if len(score) > 1 else None
            elif isinstance(score, dict):
                ft = score.get("ft")
                ht = score.get("ht")
                
                score_team1 = ft[0] if ft and len(ft) > 0 else None
                score_team2 = ft[1] if ft and len(ft) > 1 else None
                
                score_ht_team1 = ht[0] if ht and len(ht) > 0 else None
                score_ht_team2 = ht[1] if ht and len(ht) > 1 else None
            
            # Flatten match record
            flat_match = {
                "match_id": match_id,
                "tournament": tournament_name,
                "year": int(year),
                "round": match.get("round"),
                "date": match.get("date"),
                "time": match.get("time"),
                "team1": team1,
                "team2": team2,
                "score_team1": score_team1,
                "score_team2": score_team2,
                "score_ht_team1": score_ht_team1,
                "score_ht_team2": score_ht_team2,
                "group": match.get("group"),
                "ground": match.get("ground"),
            }
            all_matches.append(flat_match)
            
            # Extract goals
            for g in match.get("goals1") or []:
                all_goals.append({
                    "match_id": match_id,
                    "tournament": tournament_name,
                    "year": int(year),
                    "scorer": g.get("name"),
                    "minute": str(g.get("minute")),
                    "penalty": bool(g.get("penalty", False)),
                    "owngoal": bool(g.get("owngoal", False)),
                    "team": team1,
                    "opponent": team2
                })
            for g in match.get("goals2") or []:
                all_goals.append({
                    "match_id": match_id,
                    "tournament": tournament_name,
                    "year": int(year),
                    "scorer": g.get("name"),
                    "minute": str(g.get("minute")),
                    "penalty": bool(g.get("penalty", False)),
                    "owngoal": bool(g.get("owngoal", False)),
                    "team": team2,
                    "opponent": team1
                })
                
    return all_matches, all_goals

@dlt.source
def worldcup_source():
    matches, goals = fetch_worldcup_data()
    return [
        dlt.resource(matches, name="matches", write_disposition="replace"),
        dlt.resource(goals, name="goals", write_disposition="replace")
    ]

if __name__ == "__main__":
    db_path = os.path.abspath("worldcup.db")
    print(f"Targeting DuckDB path: {db_path}")
    
    pipeline = dlt.pipeline(
        pipeline_name="worldcup_pipeline",
        destination="duckdb",
        dataset_name="worldcup_data"
    )
    
    # Run the pipeline
    load_info = pipeline.run(worldcup_source(), credentials=db_path)
    print(load_info)
