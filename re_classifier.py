import torch

import spacy

import stanza

import spacy_stanza

from textacy import extract

# subset of https://www.clres.com/db/classes/ClassSpatial.php
# all that have a COUNT > 0 + not archaic/literary according to google translate + not in {'round', 'next door to'}
SPATIAL_PREPOSITIONS = [
    #'abaft',
    'aboard',
    'about',
    'above',
    'across',
    'after',
    'against',
    'ahead of',
    'all over',
    'along',
    'alongside',
    'amid',
    #'amidst',
    'among',
    'around',
    'as far as',
    #'astride',
    'at',
    'atop',
    'before',
    'behind',
    'below',
    'beneath',
    'beside',
    'besides',
    'between',
    #'betwixt',
    'beyond',
    'by',
    'by way of',
    #'chez',
    'down',
    #'ex',
    'for',
    'from',
    'in',
    'in front of',
    'in line with',
    'in sight of',
    'in the midst of',
    'inside',
    'inside of',
    'into',
    'near',
    'near to',
    'neath',
    #'next door to',
    #"o'er",
    'of',
    'off',
    'on',
    'on a level with',
    'on top of',
    'onto',
    'opposite',
    'out of',
    'outboard of',
    'outside',
    'outside of',
    'outwith',
    'over',
    'over against',
    'past',
    #'round',
    'round about',
    'short of',
    'this side of',
    #'thro\'',
    'through',
    'throughout',
    'to',
    'toward',
    'towards',
    'under',
    'underneath',
    'unto',
    'up',
    'up against',
    'up and down',
    'up before',
    'up to',
    'upon',
    #'via',
    'with',
    'within',
    'within sight of'
]

SPATIAL_KEYWORDS = [
    'background', 'back', 'bottom', 'backmost',
    'center', 'corner', 'close',
    'edge', 'end', 'entire',
    'facing', 'far', 'farthest', 'floor', 'foreground', 'front', 'furthest', 'frontmost',
    'ground',
    'hidden',
    'leftmost', 'left',
    'middle',
    'nearest',
    'part',
    'rightmost', 'right', 'row',
    'side',
    'top',
    'upper'
]

SPATIALS = SPATIAL_PREPOSITIONS + SPATIAL_KEYWORDS

# # Mylonas et al. (2015) The Use of English Colour Terms in Big Data
# # (...) 30 most frequent colour terms with the highest average rank across Twitter and Google Books.
# COLORS = [
#     'white', 'black', 'red', 'blue', 'brown', 'green', 'cream',
#     'yellow', 'orange', 'gray', 'grey', 'pink', 'purple', 'lime', 'olive',
#     'salmon', 'mustard', 'peach', 'coral', 'violet', 'plum', 'lavender',
#     'lilac', 'aqua', 'indigo', 'maroon', 'teal', 'turquoise', 'burgundy',
#     'aubergine', 'beige'
# ]

# 12 simple colors
COLORS = [
    'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'purple', 'pink',
    'silver', 'gold', 'beige', 'brown', 'grey', 'gray', 'black', 'white'
]


class REClassifier(object):
    def __init__(self, backend='stanza', device=None):
        assert backend in ('stanza', 'spacy')
        assert device is None or isinstance(device, torch.device)
        if backend == 'stanza':
            if device is None or device.type == 'cpu':
                use_gpu = False
            else:
                use_gpu = torch.cuda.is_available()
                torch.cuda.set_device(device)
            self.nlp = spacy_stanza.load_pipeline(
                'en',
                processors='tokenize,pos,ner',
                tokenize_no_ssplit=True,
                use_gpu=use_gpu,
            )
        else:
            #self.nlp = spacy.load('en_core_web_md')
            self.nlp = spacy.load('en_core_web_trf')

        # when checking for relational expresions, exclude nouns that refer to
        # colors or spatials
        self._EXCLUDE = COLORS + SPATIAL_KEYWORDS

    def is_spatial(self, doc):
        # check for spatial prepositions and keywords
        if isinstance(doc, str):
            doc = self.nlp(doc)
        ngrams = [w.text for w in extract.ngrams(doc, (1, 2, 3), filter_stops=False)]
        spatials = list(set(ngrams).intersection(set(SPATIALS)))
        return (spatials != [])

    def is_ordinal(self, doc):
        # check for ordinal expressions (entities)
        if isinstance(doc, str):
            doc = self.nlp(doc)
        entities = list(extract.entities(doc, include_types=('ORDINAL',)))
        return (entities != [])

    def is_relational(self, doc):
        # check if expresion is relational
        if isinstance(doc, str):
            doc = self.nlp(doc)
        valid_noun = [
            bool(w.pos_ in ('NOUN',) and w.text not in self._EXCLUDE)
            for w in doc
        ]
        relationals = [
            w.text
            for w in doc
            if w.pos_ == 'ADP'  # adposition
            and any(valid_noun[:w.i])  # a noun to the left of the ADP
            and any(valid_noun[w.i+1:])  # a noun to the right of the ADP
        ]

        return (relationals != [])

    def classify(self, expr):
        doc = self.nlp(expr)
        is_spatial = int(self.is_spatial(doc))
        is_ordinal = int(self.is_ordinal(doc))
        is_relational = int(self.is_relational(doc))
        return (is_spatial, is_ordinal, is_relational)


def main():
    data_root, dataset, split_by = 'refer/data', 'refclef', 'berkeley'
    #data_root, dataset, split_by = 'refer/data', 'refcoco', 'unc'
    #data_root, dataset, split_by = 'refer/data', 'refcocog', 'umd'

    import os
    import sys
    sys.path.append('refer')
    from refer import REFER
    refer = REFER(data_root, dataset, split_by)

    classifier = REClassifier(backend='stanza')
    print()

    all_, intrinsic, spatial, ordinal, relational = [], [], [], [], []

    for rid in refer.getRefIds(split='test'):
        ref = refer.Refs[rid]
        ann = refer.refToAnn[rid]

        file_name = refer.Imgs[ref['image_id']]['file_name']
        if dataset == 'refclef':
            file_name = os.path.join(
                'refer', 'data', 'images', 'saiapr_tc-12', file_name
            )
        else:
            coco_set = file_name.split('_')[1]
            file_name = os.path.join(
                'refer', 'data', 'images', 'mscoco', coco_set, file_name
            )

        sentences = [s['sent'] for s in ref['sentences']]

        for i, sent in enumerate(sentences):
            stype = classifier.classify(sent)

            len_ = len(sent.split())

            all_.append(len_)
            if sum(stype) == 0:
                intrinsic.append(len_)
            if stype[0]:
                spatial.append(len_)
            if stype[1]:
                ordinal.append(len_)
            if stype[2]:
                relational.append(len_)

            # print(*stype, sent)#, [(w, w.pos_) for w in doc])

    print(f'all: {len(all_)}, {np.mean(all_):.2f} ({np.std(all_):.2f})')
    print(f'intrinsic: {len(intrinsic)}, {np.mean(intrinsic):.2f} ({np.std(intrinsic):.2f})')
    print(f'spatial: {len(spatial)}, {np.mean(spatial):.2f} ({np.std(spatial):.2f})')
    print(f'ordinal: {len(ordinal)}, {np.mean(ordinal):.2f} ({np.std(ordinal):.2f})')
    print(f'relational: {len(relational)}, {np.mean(relational):.2f} ({np.std(relational):.2f})')


if __name__ == '__main__':
    main()
