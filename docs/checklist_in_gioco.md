# Checklist di controllo in gioco

Mondo di prova: `mappa_arda.png`, lato 1000, seed predefinito (`esempi/genera_mappa.py --lato 1000`).
Le coordinate sono di gioco (x, z) e valgono per quel mondo; con un'altra mappa o altre
opzioni cambiano. Per ricavarle di nuovo: pianificare con le stesse opzioni e sottrarre
mezzo lato (500) alle coordinate di mappa.

Teletrasporto (modalita' creativa): `/tp @s <x> <y+6> <z>`. Per ogni punto: guarda, scatta
uno screenshot (F2) e segna **OK** o **difetto** con una riga.

## 1. Città murate (le quattro più grandi)
Cosa guardare: pianta ordinata; cinta chiusa con 4 porte e torri; fosso pieno e senza
trabocchi; **ponte alla quota della strada** (nessun gradino di un blocco); piazza con
campana, fontana e 4 banchi; case una per lotto, tutte diverse.

| Città | Centro (x, z) | Quota | Punto di arrivo |
|---|---|---|---|
| A | 321, -147 | 79 | `/tp @s 321 85 -147` |
| B | 239, -322 | 78 | `/tp @s 239 84 -322` |
| C | 318, -26 | 79 | `/tp @s 318 85 -26` |
| D | 155, -88 | 79 | `/tp @s 155 85 -88` |

Ponti sul fosso (porta est, poi le altre): A `377 85 -147`, B `295 85 -322`, C `366 66 -26`,
D `211 85 -88`.
- **C, porta est (366, -26): la strada esterna sta a quota 60, la città a 79.** Guarda se
  scende in modo decente o se è una parete.
- Il verificatore automatico segnala un pezzo sospeso vicino a (314, 79, -77): ponte della città C?

## 2. Villaggi
| Villaggio | Centro (x, z) | Lotti | Punto |
|---|---|---|---|
| E | -59, 59 | 12 | `/tp @s -59 85 59` |
| F | 50, -407 | 12 | `/tp @s 50 85 -407` |
| G | 331, -279 | **4** | `/tp @s 331 85 -279` |

G è il caso limite: quattro case. Dimmi se e' accettabile o se alzare il minimo.

## 3. Banchi di mercato (non sospesi, sulla piazza)
Città A: `/tp @s 320 85 -148`; città B: `/tp @s 238 84 -323`. Guarda sotto i banchi
(pali fino a terra) e che i venditori siano dietro il bancone.

## 4. Miniere (22 nel mondo di prova)
Cosa guardare: portale ad arco di legno e pietra; binario e ghiaia davanti; trincea con
telai e lanterne; discesa a 1 blocco ogni 3 celle; carrelli; minerali piu' ricchi in
fondo (lapislazzuli, oro, diamanti a y<20).

| N | Portale (x, z, y) | Fondo y | Piani | Perché |
|---|---|---|---|---|
| 1 | -304, -76, 67 | 0 | 3 | la più profonda |
| 11 | -303, -170, 67 | -5 | 3 | la più profonda, diamanti |
| 5 | -41, -81, 69 | 0 | 3 | |
| 4 | -264, 141, 80 | 60 | 1 | una superficiale (solo carbone/rame) |

Punto: `/tp @s <x> <y+6> <z>`. Scendi a piedi fino in fondo e conta i piani.

## 5. Case isolate (7)
`/tp @s -209 78 -301`, `-172 80 -241`, `-125 77 -170`, `-342 77 -21`, `4 76 -286`,
`34 77 -348`. Guarda: casa appoggiata a terra (niente vuoto sotto), nessun albero dentro.

## 6. Avamposti, cimiteri, portali (12)
Portale: `/tp @s 4 85 -202`; cimitero: `/tp @s 455 85 254`; campo: `/tp @s -124 85 -76`.
Guarda: struttura dal template, nemici (scheletri/zombie) solo dentro.

## 7. Arredi di città
Fontana A: `/tp @s 319 85 -157`; campana: `/tp @s 322 85 -147`; lampioni lungo
l'asse est: `/tp @s 271 85 -147`; recinto: `/tp @s 287 85 -7`; giardino: `/tp @s 287 85 -42`.

## 8. Acqua
Il verificatore trova ancora ~4400 celle d'acqua con aria accanto, per lo più fiumi in quota.
Punti: `/tp @s -451 70 -140` e `/tp @s -388 92 -427`. Guarda se l'acqua cola sulle sponde.

## Come riportare
Un elenco "punto → OK / difetto + coordinate + screenshot". Ogni difetto diventa una issue.
