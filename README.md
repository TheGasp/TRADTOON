
# TRADTOON

Ce projet dévellopé en python a pour vocation de permettre de facilement traduire des chapitres de BD, Webtoon ou Manga.

Toutes les langues disponibles dans tesseract sont utilisables bien que les performances de certaines (tels que le japonais, le chinois ou le coréen) soient inférieurs avec le programme.


## Demonstration

Voila ci dessous un exemple d'une traduction de Webtoon de l'anglais vers le francais.

![Image3](https://github.com/user-attachments/assets/d6672ce5-9a5a-44e6-ad7b-b01a73174ee8)




## API et Bibliothèques nécessaires

#### API

| Parametre | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `API_KEY_TRAD ` | `string` | **Required**. Votre clés pour l'API de traduction DeepL |

#### Bibliothèques

| Nom  | Description                       |
| :-------- | :-------------------------------- |
| `cv2`      | **Required**. Bibliotèque nécessaires pour le traitement et la gestion des images (aussi appelée OpenCV) |
| `numpy`      | **Required**. Utilisation en parralélle de OpenCV |
| `pytesseract`      | **Required**. Bibliothèque s'occupant de l'OCR |
| `deepl`      | **Required**. Bibliothèque s'occupant de la traduction (fonctionne aussi avec les bibliothèques de Google ou autre)|


## Détails techniques

### Extraction du texte

Pour réaliser cette étape, on s'appuie dans un premier temps sur un filtrage de l'image de manière à pouvoir plus aisément détecter le texte. On applique ensuite pytesseract et l'on trie ce qui est obtenu de manière à supprimer les doublons ou les faux positifs.

### Traduction du texte

On vient dans un premier temps nettoyer le texte obtenu, de manière à homogénéiser le résultat, puis on le fait traduire par l'API de DeepL.

### Remplacement du texte
Dans cette étape, on vient calculer la taille de la police à utiliser pour que le texte nouvellement traduit rentre dans la zone attribuée. On détermine aussi la couleur du texte et du fond de manière à ce que le nouveau texte s'intègre au mieux dans la page.
