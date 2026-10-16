# Checklist di controllo in gioco - mappa_arda

Per ogni punto: `/tp @s x y z` in creativa o spettatore, guarda, segna OK o difetto.

## Torri con un capo

Torre di pietra a quattro piani: porta, scala a pioli, capo con nome in cima, due guardie, due forzieri; cartello davanti.

- [ ] **Torre di Aldric il Cupo** - `/tp @s -338 75 -110` (capo: Aldric il Cupo)
- [ ] **Torre di Corvo Nero** - `/tp @s 452 87 210` (capo: Corvo Nero)
- [ ] **Torre di Garrick Senzaterra** - `/tp @s 83 84 281` (capo: Garrick Senzaterra)
- [ ] **Torre di Ivor il Custode** - `/tp @s -175 77 -319` (capo: Ivor il Custode)
- [ ] **Torre di Kael Lamarotta** - `/tp @s 43 75 -227` (capo: Kael Lamarotta)

## Incarichi

Un abitante con il nome sopra la testa, su una strada vicino alla piazza. Per 1 smeraldo vende un libro con l'incarico. Il capo (uccidi) lascia cadere l'oggetto, la reliquia (recupera) sta nel forziere del tesoro, la lettera (salva) nella cella del castello. Lo scambio di consegna vale solo con QUELL'oggetto (un lingotto qualunque no) e paga smeraldi o diamanti. Controlla che il committente non perda la professione.

- [ ] **Notaio Corvino** (Rivabella) - `/tp @s 321 100 -147` - recupera a Tomba del Re Smarrito (`/tp @s 219 100 -237`), oggetto «Reliquia di Tomba del Re Smarrito», ricompensa 18 emerald
- [ ] **Dama Fiammetta** (Pietrafonda) - `/tp @s 259 100 -318` - recupera a Cripta dell'Osso (`/tp @s 43 100 -334`), oggetto «Reliquia di Cripta dell'Osso», ricompensa 22 emerald
- [ ] **Scriba Almerico** (Villabianca) - `/tp @s 318 100 -6` - uccidi a Torre di Corvo Nero (`/tp @s 452 100 210`), oggetto «Sigillo di Torre di Corvo Nero», ricompensa 24 emerald
- [ ] **Cancelliere Orsino** (Granvalle) - `/tp @s 155 100 -88` - uccidi a Torre di Kael Lamarotta (`/tp @s 43 100 -227`), oggetto «Sigillo di Torre di Kael Lamarotta», ricompensa 20 emerald
- [ ] **Dama Berenice** (Valdargento) - `/tp @s 363 100 -247` - uccidi a Fortezza di Signore di Roccanera (`/tp @s -128 100 -74`), oggetto «Sigillo di Fortezza di Signore di Roccanera», ricompensa 6 diamond
- [ ] **Madonna Isotta** (Monteverde) - `/tp @s -59 100 59` - recupera a Rovine di Altomuro (`/tp @s -207 100 -78`), oggetto «Reliquia di Rovine di Altomuro», ricompensa 22 emerald
- [ ] **Podesta' Tebaldo** (Colledoro) - `/tp @s 50 100 -407` - uccidi a Torre di Ivor il Custode (`/tp @s -175 100 -319`), oggetto «Sigillo di Torre di Ivor il Custode», ricompensa 24 emerald

## Cascate

Dove un fiume scende di 3+ blocchi: colonna d'acqua in caduta con pareti di roccia ai lati. Controlla che l'acqua cada senza spargersi sul prato, che la pozza non si svuoti e che a valle il fiume prosegua. Cinque dei salti piu' alti e distanti fra loro.

- [ ] salto di 9 blocchi - `/tp @s 129 79 -428`
- [ ] salto di 9 blocchi - `/tp @s -356 79 -425`
- [ ] salto di 9 blocchi - `/tp @s 195 79 -75`
- [ ] salto di 9 blocchi - `/tp @s 24 79 67`
- [ ] salto di 9 blocchi - `/tp @s -87 79 147`

## Altro da controllare

- [ ] Grotte lussureggianti: oltre alle pozze, ruscelli sinuosi di acqua ferma in un solco di una cella, chiusi dalla roccia (non scorrono e non si svuotano); si interrompono ai gradini del pavimento.
- [ ] Giungla: ogni tanto una pianta brocca (`pitcher_plant`), alta due blocchi.
- [ ] Erba secca (savana, deserto) e lettiera di foglie sotto gli alberi dei boschi e del bosco scuro.
- [ ] Lampioni di rame veri (non bulbi) nelle citta', di ossidazione diversa da uno all'altro.
- [ ] I moli di legno procedurali (se ci sono) hanno il legno del bioma, non sempre abete.
- [ ] Il tetto delle torri ha un parafulmine di rame al centro.
- [ ] I forzieri di nave, faro e castello da template hanno il loro bottino (oro e provviste di bordo; carbone, torce, canna da pesca; armi, libri, mele d'oro), non la dispensa di una casa.
- [ ] Le costruzioni di arenaria (se ce n'e') hanno blocchi lisci, tagliati e scolpiti mescolati.
- [ ] Gli spawner delle cripte fanno uscire ogni tanto un mob diverso (zombie con scheletro, ragno...).
- [ ] Nel log all'apertura del mondo non devono comparire 'Failed to load chunk'.

## Mappe del tesoro

Nel forziere del tesoro di ogni rovina, torre, fortezza e castello, accanto al libro, c'e' una mappa (`Mappa: <luogo>`) del paesaggio attorno alla tappa successiva, con una croce rossa sul posto e un puntino bianco dov'e' il forziere. Dopo aver aperto il forziere:

- [ ] la mappa si vede (non e' bianca) e mostra terra, mare, montagne e la croce rossa al centro;
- [ ] aprendola nel mondo la croce cade sul luogo giusto (la descrizione dice quale);
- [ ] il nome e la descrizione dell'oggetto si leggono e non si aggiorna camminando (e' bloccata);
- [ ] se e' bianca o dice 'mappa sconosciuta': scrivimelo, il formato del file di mappa in 26.x non e' stato provato (`data/minecraft/maps/<n>.dat`).

## Castelli in rovina

Cinta con brecce e torri spezzate, portale sfondato, cortile invaso da erba, macerie, ossa, alberi morti e tombe; il maschio sventrato con trono (il capo siede li'), tavola rotta, scala rovinata; torre dello spawner (SW). Dalla sala, a sinistra, una scala scende alla cripta: anticamera, DUE fili d'inciampo con distributori di frecce, fossa con le punte (sponda stretta su un lato), sala dello spawner, celle con prigionieri, caveau con piastre sul TNT, forzieri-esca e il tesoro. Il cartello e' fuori dal portale.

- [ ] **Rocca delle Ombre** - `/tp @s -316 75 -152` (in rovina, signore: Il Castellano Senzavolto)

## Fortezze

Cinta di pietra 29x29 con quattro torri d'angolo (scala a pioli, arciere sul tetto), portale con saracinesca alzata, camminamento con merli (scale d'accesso sul lato di fondo), cortile con pozzo, fucina con cassa, stalla, bracieri, lampioni e stendardi; il maschio al centro e' una torre curata con il capo (nome, armatura) e quattro soldati nel cortile.

- [ ] **Fortezza di Signore di Roccanera** - `/tp @s -128 74 -74` (capo: Signore di Roccanera)

## Rovine con cripta

Rovina in superficie, scala di 14 blocchi, anticamera, sala dello spawner, tesoro con due forzieri-trappola sul TNT. Cartello davanti. Lo spawner sta circa 41 blocchi dalla rovina.

- [ ] **Cripta dell'Osso** - `/tp @s 43 78 -334` (cripta con skeleton)
- [ ] **Cripta della Veglia** - `/tp @s -425 78 -20` (cripta con skeleton)
- [ ] **Cripta di Orrenda** - `/tp @s -237 74 -357` (cripta con zombie)
- [ ] **Rovine di Altomuro** - `/tp @s -207 74 -78` (cripta con skeleton)
- [ ] **Sepolcro dei Cento** - `/tp @s 43 86 286` (cripta con zombie)
- [ ] **Tomba del Re Smarrito** - `/tp @s 219 74 -237` (cripta con skeleton)

## Miniere

Imbocco con cartello, collina del posto, binari. Nell'ultimo piano: camera finale con pozza.

- [ ] **Miniera dei Due Fratelli** - `/tp @s -206 76 -302`
- [ ] **Miniera dei Sospiri** - `/tp @s 421 83 312`
- [ ] **Miniera dei Tre Pozzi** - `/tp @s 48 91 -465`
- [ ] **Miniera del Carbone** - `/tp @s -145 76 -3`
- [ ] **Miniera del Cinghiale** - `/tp @s 111 84 313`
- [ ] **Miniera del Corvo** - `/tp @s -39 78 263`
- [ ] **Miniera del Cristallo** - `/tp @s -296 84 138`
- [ ] **Miniera del Falco** - `/tp @s 321 80 -399`
- [ ] **Miniera del Ghiaccio** - `/tp @s -420 75 -167`
- [ ] **Miniera del Gufo** - `/tp @s -100 84 166`
- [ ] **Miniera del Re Cieco** - `/tp @s 244 77 -164`
- [ ] **Miniera del Sale** - `/tp @s -294 77 -257`
- [ ] **Miniera del Salice** - `/tp @s -390 82 -326`
- [ ] **Miniera del Vento** - `/tp @s -111 78 -153`
- [ ] **Miniera dell'Argento Nero** - `/tp @s -285 102 -412`
- [ ] **Miniera dell'Ultima Vena** - `/tp @s 358 67 -316`
- [ ] **Miniera della Lanterna** - `/tp @s 11 73 -235`
- [ ] **Miniera della Luna Cava** - `/tp @s -250 84 285`
- [ ] **Miniera della Pioggia** - `/tp @s 212 85 299`
- [ ] **Miniera dello Stagno** - `/tp @s -304 73 -76`
- [ ] **Miniera di Ferrovecchio** - `/tp @s -191 85 127`
- [ ] **Miniera di Pietrabuia** - `/tp @s 111 83 -362`

## Citta'

Piazza con fontana (parafulmine), campana, 2 stendardi, 2 bracieri, lampioni di rame; giardini con zucche e alveari; legno degli arredi del bioma; gatti sulle strade; un libro 'Il diario del viandante' in alcune case della citta' piu' vicina al centro.

- [ ] **Colledoro** - `/tp @s 50 86 -407` (raggio 41)
- [ ] **Granvalle** - `/tp @s 155 86 -88` (raggio 54, murata, tondo)
- [ ] **Monteverde** - `/tp @s -59 86 59` (raggio 41)
- [ ] **Pietrafonda** - `/tp @s 259 86 -318` (raggio 54, murata, tondo)
- [ ] **Rivabella** - `/tp @s 321 86 -147` (raggio 54, murata)
- [ ] **Valdargento** - `/tp @s 363 86 -247` (raggio 33)
- [ ] **Villabianca** - `/tp @s 318 86 -6` (raggio 46, murata, tondo)

## Accampamenti

Cerchio calpestato, falo', predone con balestra, cavalli.

- [ ] **Accampamento del Lupo** - `/tp @s 163 83 261`
- [ ] **Accampamento delle Ceneri** - `/tp @s 180 77 -316`
- [ ] **Bivacco dell'Esule** - `/tp @s 4 85 -201`
- [ ] **Campo dei Predoni** - `/tp @s -250 74 -91`
- [ ] **Campo dei Senzanome** - `/tp @s -65 80 -319`
- [ ] **Campo dei Senzanome 3** - `/tp @s -221 82 -283`
- [ ] **Campo del Corvo** - `/tp @s 484 85 316`

## Cimiteri

Forzieri con ossa e refurtiva; scheletro (stray sulla neve).

- [ ] **Camposanto dei Dimenticati** - `/tp @s 460 84 154`
- [ ] **Cimitero degli Impiccati** - `/tp @s -377 85 -284`
- [ ] **Cimitero del Silenzio** - `/tp @s 106 92 -375`

## Biomi

- [ ] **badlands** (17432 celle) - `/tp @s -27 88 -214` - sabbia rossa e fasce di terracotta
- [ ] **mangrove_swamp** (11241 celle) - `/tp @s -37 74 -59` - mangrovie con radici, fango, rane
- [ ] **savanna** (30435 celle) - `/tp @s 120 85 328` - acacie piatte, lama, asini
- [ ] **jungle** (1094 celle) - `/tp @s 245 80 331` - alberi altissimi, bambu', liane, panda
- [ ] **taiga** (2009 celle) - `/tp @s 23 88 -404` - abeti, felci grandi
- [ ] **snowy_tundra** (4843 celle) - `/tp @s -172 102 55` - neve sottile sugli alberi, stray
- [ ] **dark_forest** (1216 celle) - `/tp @s 41 81 78` - rovere scuro, funghi giganti
- [ ] **meadow** (99486 celle) - `/tp @s 6 77 -323` - ciliegi, lilla', rose, peonie, api
- [ ] **sunflower_plains** (37769 celle) - `/tp @s -12 83 30` - girasoli
- [ ] **warm_ocean** (131202 celle) - `/tp @s 6 46 402` - coralli e pesci tropicali
- [ ] **desert** (90367 celle) - `/tp @s -167 75 -153` - cactus, cammelli, husk negli accampamenti

## Bordo della mappa

- [ ] Vola oltre un bordo (es. `/tp @s 480 120 0` e poi verso est): il terreno continua e si raccorda al nostro senza un taglio netto.

## Catena degli indizi

- [ ] In una casa della citta' piu' vicina al centro: libro che indica la prima tappa. Seguila: il tesoro di ogni rovina/torre ha un libro che indica la successiva.
- [ ] Un cartografo o un libraio vende per 5 smeraldi un libro con la rovina o la torre piu' vicina.
- [ ] Cartelli ai bivi con i due luoghi piu' vicini e la direzione.

## Sottoterra

- [ ] Grotte con dripstone, muschio, pozze, sculk profondo, geodi di ametista (vedi `docs/`).
