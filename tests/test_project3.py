import project3
import os

def test_process_pdf():
    text, name = project3.process_pdf('testcity.pdf')
    assert type(text) == str and type(name) == str


def test_normalize_corpus():
    text = '<div> welcome and in of to ;; walrus'
    clean = project3.normalize_corpus(text)
    assert clean == 'welcome walrus'

def test_predict_text():
    text = 'transporation urban neighborhood'
    cluster = project3.predict_text(list(text))[0]
    assert cluster in range(0,1)

def test_output():
    row = ['townsville','the dog rides the bus','dog rides bus',0]
    filename = 'example_tsv.tsv'
    output_text = project3.output(row,filename)
    test1 = (output_text == '[townsville] clusterid: 0')
    test2 = (os.path.isfile(filename))
    os.unlink(filename)
    assert test1 and test2
