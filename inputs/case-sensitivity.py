# Dit script zet de hoofdletters goed. 
# dit moet nog vervangen worden door script van Deltares, maar deze versie kopieert gewoon alles. 
# Als alles er op hoofdletter en niet hoofdletter manier in staat werkt het namelijk ook
import shutil
import os

path = r"/data/rr"

for file in os.listdir(path):
    old = os.path.join(path, file)
    new = os.path.join(path, file.lower())
    try:
        shutil.copyfile(old, new)
    except:
        print("{} and {} did not copy, probably because they are the same file.".format(new, old))

