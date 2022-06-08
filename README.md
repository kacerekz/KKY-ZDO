# ZDO2022

Semetrální práce z předmětu [ZDO v roce 2022](https://nbviewer.jupyter.org/github/mjirik/ZDO/blob/master/ZDOsem2022.ipynb)

# Spuštění
Hlavní implementovaný algoritmus lze spustit pomocí dodaného skriptu s automatickým testem jako

```shell
python test_zdo.py
```

přičemž vstupní data jsou očekávána v adresáři tests/test_dataset nebo na cestě dané systémovou proměnnou prostředí ZDO_DATA_PATH. Skript ukládá výstup do složky results do souborů out-<filename>.avi a  out-<filename>.json. Tedy např. out-5.mp4.avi.

Skript zdo_sp2.py lze spustit s parametrem path:

```shell
python zdo_sp2.py <path>
```
  
který představuje absolutní či relativní cestu k testovanému .mp4 souboru. Skript ukládá výstup do složky results do souborů out2.avi a out2.json.
