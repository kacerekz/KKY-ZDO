# ZDO2022

Semetrální práce z předmětu [ZDO v roce 2022](https://nbviewer.jupyter.org/github/mjirik/ZDO/blob/master/ZDOsem2022.ipynb)

# Spuštění

Skript zdo_sp.py lze spustit příkazem ve tvaru

```shell
python zdo_sp.py <dir> <filter> <interpolate>
```
kde dir je cesta ve tvaru např. "D:/Data/ZDO/224" která obsahuje adresář "images" a soubor annotations.xml. Parametr filter pak musí nabývat hodnotu 1 nebo 2, kdy pro 1 provádí základní filtraci a pro 2 provádí filtraci s využitím algoritmu k-means. Pro parametr interpolate 1 je interpolace zapnutá, pro parametr 2 je vypnutá.

Skript zdo_sp2.py lze spustit pouze s parametrem dir, více parametrů neočekává.