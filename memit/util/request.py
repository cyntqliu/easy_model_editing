'''
Utility file for generating additional, evaluation-specific parts of the MEMIT request
'''
import os
import sys
from typing import Dict, List

import jsonlines
from requests import get
from SPARQLWrapper import SPARQLWrapper, JSON

sys.path.append('../../')


# ========================== NEIGHBORHOOD (WikiData) ==========================
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


# ======================= PARAPHRASE (ParaRel + Parrot) =======================
def process_pararel_patterns(
    paraphrases: List[str],
    subject: str,
    target: str,
) -> List[str]:
    '''
    Process the patterns from ParaRel to a format matching the prompts expected
    by MEMIT by:
    
    1. Filtering out patterns whose targets appear before the subjects, as this is
    currently unsupported by MEMIT
    2. Replacing the subject indicator with `{}`
    3. Removing everything after the target indicator
    4. Removing duplicates resulting from step 3
    '''
    reformatted_paraphrases = []
    for p in paraphrases:
        if p.index(target) > p.index(subject):
            pre_target_str = p[:p.index(target)].strip()
            pre_target_str = pre_target_str.replace(subject, "{}")
            # after processing, the paraphrase may be identical to a previous one
            if pre_target_str not in reformatted_paraphrases:
                reformatted_paraphrases.append(pre_target_str)

    return reformatted_paraphrases


def get_ai_paraphrases(request: Dict, max_sentences: int) -> List[str]:
    '''
    Given a request, use an NLU library to generate up to max_sentences paraphrases
    '''
    return []


def get_paraphrase(request: Dict, max_sentences: int=10, use_nlp: bool=True) -> List[str]:
    '''
    Given a request with keys "prompt", "target_new", and "subject",
    generating paraphrased sentences with the same subject and target_new,
    but rewordings of the prompt that have essentially the same meaning

    These sentences will be used to test the generality of the edit.

    Args:
        request: Dict[str], the edit request from which we want to generate paraphrases
        max_sentences: int, the maximum number of paraphrases to generate
        use_nlp: bool, whether to use an NLP library to supplement the ParaRel paraphrases.
            Will be slower, but will not be limited to the fixed ParaRel database.
            Default: `True`

    returns: List[str], empty if no paraphrases were found
    '''
    pararel_dir = "pararel/"
    subject_indicator = "[X]"
    target_indicator = "[Y]"
    paraphrases = []
    for f in os.listdir(pararel_dir):
        if f.endswith('.jsonl'):
            with jsonlines.open(os.path.join(pararel_dir, f)) as rdr:
                all_paraphrases_f = [pattern["pattern"] for pattern in rdr]

                # process patterns so they match the request format
                all_paraphrases_f = process_pararel_patterns(
                    all_paraphrases_f, subject_indicator, target_indicator,
                )

                if request["prompt"] in all_paraphrases_f:
                    paraphrases.extend([p for p in all_paraphrases_f if p != request["prompt"]])

    paraphrases = paraphrases[:max_sentences]
    if use_nlp and len(paraphrases) < max_sentences:
        paraphrases.extend(get_ai_paraphrases(request, max_sentences-len(paraphrases)))

    return paraphrases
