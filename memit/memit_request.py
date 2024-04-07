'''
Main code for getting requests in the format needed by MEMIT
'''
from util.request import get_neighborhood, get_paraphrase


PARAPHRASE_KEY = "paraphrase_prompts"
NEIGHBORHOOD_KEY = "neighborhood_prompts"


def extract_entity_and_relations(doc, **kwargs):
    '''
    Given a str document `doc`, extract entities and their associated
    relations from the document, then reprocess them into the format needed
    for MEMIT

    returns: List[Dict[key, any]] in which the Dict must contain "prompt",
        "subject", and "target_new" keys. TODO: how do we get "target_true"
        from the model?
    '''
    pass

def get_request(docs, whole_file=False, **kwargs):
    '''
    Builds the request object used for model editing, as well as
    paraphrase and neighborhood prompts for eval. An example request is

    request = {
        "prompt": "{} plays the sport of", // relation
        "subject": "LeBron James", // to be inserted into prompt
        "target_new": {
                "str": "football"
        },
        "target_true": {
            "str": …
        },
    }

    where `request` is used in the following structure:
    
    example = {
        "case_id": int,
        "requested_rewrite": request,
        "paraphrase_prompts": [...], // list of paraphrase prompts, in `eval_prompts`
        "neighborhood_prompts": [...], // list of neighborhood prompts, in `eval_prompts`
        "generation_prompts": [...], // list of generation prompts
    }
    '''
    if not whole_file:
        # line separated
        request = []
        for doc in docs:
            doc_request = extract_entity_and_relations(doc, **kwargs)
            request.extend(doc_request)
            
    else:
        request = extract_entity_and_relations(docs, **kwargs)

    # add neighboring and paraphrase sentences
    eval_prompts = []
    for req in request:
        req_prompts = dict()
        para_sentences = get_paraphrase(req)
        if len(para_sentences):
            req_prompts[PARAPHRASE_KEY] = para_sentences

        nbhd_sentences = get_neighborhood(req)
        if len(nbhd_sentences):
            req_prompts[NEIGHBORHOOD_KEY] = nbhd_sentences

        eval_prompts.append(req_prompts)
    
    return request, eval_prompts
