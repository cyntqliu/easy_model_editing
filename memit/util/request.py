'''
Utility file for generating additional, evaluation-specific parts of the MEMIT request
'''
import sys
from typing import Dict, List

from requests import get
from SPARQLWrapper import SPARQLWrapper, JSON


endpoint_url = "https://query.wikidata.org/sparql"


def get_qnumber(wikiarticle: str) -> str:
    # get the q number of the topic of wikiarticle
    # code from https://stackoverflow.com/questions/56281527/finding-wikidata-identifiers-properties-lexemes
    resp = get('https://www.wikidata.org/w/api.php', {
        'action': 'wbgetentities',
        'titles': wikiarticle,
        'sites': 'enwiki',
        'format': 'json'
    }).json()

    id_list = list(resp['entities'].keys())
    if len(id_list):
        return id_list[0]
    
    return None


def get_pnumber(subject: str, object: str) -> str:
    # get the p number of the relation
    # because querying relations depends on exact wording, we instead search for a relation
    # fulfilling subject (relation) object and return its P value
    query = '''SELECT ?relation
    WHERE 
    {{
      wd:{0} ?relation wd:{1}    
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 1'''.format(subject, object)

    results = get_wikidata_results(endpoint_url, query)
    if len(results["results"]["bindings"]):
        p_url = results["results"]["bindings"][0]["relation"]["value"]
        return p_url.split('/')[-1]
    
    return None


def get_wikidata_results(endpoint_url: str, query: str) -> Dict[str, any]:
    user_agent = "easy-model-editing/%s.%s (cynliu98@alum.mit.edu)" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_neighborhood(request: Dict, max_sentences: int=10) -> List[str]:
    '''
    Given a request with keys "prompt", "target_true", and "subject",
    generating neighborhood sentences with the same prompt and target_true,
    but different subjects.

    These sentences will be used to test the specificity of the edit.

    returns: List[str], empty if no neighboring sentences were found
    '''
    neighborhood_sentences = []
    q_number_target = get_qnumber(request["target_true"]["str"])
    q_number_subject = get_qnumber(request["subject"])
    if q_number_subject is None or q_number_target is None:
        return neighborhood_sentences
    
    p_number = get_pnumber(q_number_subject, q_number_target)
    if p_number is None:
        return neighborhood_sentences
    
    query = '''SELECT ?itemLabel
    WHERE 
    {{
      ?item wdt:{0} wd:{1}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {2}'''.format(p_number, q_number_target, max_sentences)

    results = get_wikidata_results(endpoint_url, query)
    for result in results["results"]["bindings"]:
        neighborhood_sentences.append(
            request['prompt'].format(result['itemLabel']['value'])
        )

    return neighborhood_sentences


def get_paraphrase(request):
    '''
    Given a request with keys "prompt", "target_new", and "subject",
    generating paraphrased sentences with the same subject and target_new,
    but rewordings of the prompt that have essentially the same meaning

    These sentences will be used to test the generality of the edit.

    returns: List[str], empty if no paraphrases were found
    '''
    pass
