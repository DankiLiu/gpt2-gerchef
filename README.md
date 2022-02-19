# gpt2-gerchef
A model fine-tuned from GPT-2 to generate German recipes

19.02.22
The model can generate recipes after 3 epochs of training, using train batch size of 8 and evaluation batch size of 16.
An example is>
Zuerst Blumenkohl (sollte eigentlich nicht sein) dazu geben und alles mit ein bis zwei Esslöffeln Wasser und Mehl zu einem glatten Teig kneten bis der Teig aufgebraucht ist. Den Teig zu einer Kugel formen und auf ein Backblech legen

Problem:
Generated recipe can not end properly. Output length only depend on the #max_length# config.
