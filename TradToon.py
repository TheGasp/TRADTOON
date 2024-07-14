import cv2
import numpy as np
import pytesseract
import deepl
import os

#Initialisation
translator = deepl.Translator("Votre clé")
font = cv2.FONT_HERSHEY_SIMPLEX #police de caractére
inter_espace = 0.25 #pourcentage de la hauteur de la ligne entre chaque ligne
font_thickness = 2 #Valeur de calcul pas forcement de representation

config = r'--psm 6 '
langue = 'eng'

#Traitement des images
def treat_image(image):
    #Detection du text
    image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient = cv2.morphologyEx(image_gris, cv2.MORPH_GRADIENT, np.ones((1, 9), np.uint8))# Calculer le gradient vertical

    #Binariser l'image des gradients en utilisant la valeur d'entropie
    _, binarized = cv2.threshold(gradient, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Fermeture verticale
    closed_v = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    # Fermeture horizontale
    closed_h = cv2.morphologyEx(closed_v, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))

    vertical_element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))  # 1 pixel de large, 4 pixels de haut (fermeture verticale)

    #Appliquer la dilatation verticale
    closed_image = cv2.dilate(closed_h, vertical_element, iterations=2)

    #Trouver les contours dans l'image
    contours = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = contours[::-1] #Inversion de la liste

    return contours


#Gestion du background
def median_color(rectangle, image):
    x1, y1, x2, y2 = rectangle # Extraire les coordonnées du rectangle

    # Récupérer les pixels du contour du rectangle
    rectangle_pixels = []
    for x in range(x1, x2 + 1, 2): #Pas = 2
        rectangle_pixels.append(image[y1, x])
        rectangle_pixels.append(image[y2, x])
    for y in range(y1 + 1, y2, 2):
        rectangle_pixels.append(image[y, x1])
        rectangle_pixels.append(image[y, x2])

    # Convertir la liste de pixels en un tableau NumPy
    rectangle_pixels = np.array(rectangle_pixels)

    # Calculer la médiane des couleurs BGR
    median_color_bgr = np.median(rectangle_pixels, axis=0).astype(np.uint8)
    median_color_bgr = (int(median_color_bgr[0]), int(median_color_bgr[1]), int(median_color_bgr[2]))

    return median_color_bgr

#Gestion de la couleur de la police d'ecriture
def complementary_color(color):
    # Couleur blanche et noire en RGB
    blanc = (255, 255, 255)
    noir = (0, 0, 0)

    # Calcul de la distance euclidienne aux couleurs blanche et noire
    distance_blanc = sum((c1 - c2) ** 2 for c1, c2 in zip(color, blanc))
    distance_noir = sum((c1 - c2) ** 2 for c1, c2 in zip(color, noir))

    # Retourne la couleur contrastante en fonction de la distance
    return noir if distance_blanc < distance_noir else blanc

#Verification des bulles
def check_bubble(chaine):
    if langue == 'eng':#Test valide pour les langues latines
        ponctuation = ",'!?." # Définir une liste de signes de ponctuation
        chaine_sans_ponctuation = ''.join(char for char in chaine if char not in ponctuation)

        # Compter le nombre de lettres majuscules dans la chaîne sans ponctuation
        nb_majuscules = sum(1 for char in chaine_sans_ponctuation if char.isupper())

        if len(chaine_sans_ponctuation) > 0 :
            proportion_maj = nb_majuscules / len(chaine_sans_ponctuation)

            if proportion_maj > 0.50 :
                return True

        return False

    return True

#Nettoyage pre traduction
def pre_clean(text):
    text = text.replace("\n", " ") #Enleve les sauts de ligne
    char_banned = [("/\;:1|[]{}", "i"),("2","?"),("5$","s"),("0","o"),("»«","..."),("7","T"),("-¥+","")]

    for i in range(len(char_banned)):
        for char in char_banned[i][0]:
            text = text.replace(char, char_banned[i][1])

    return text.lower()


#Groupement des contours
def group_text_into_line(contours):
    dialogue_line = []

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        contour_area = cv2.contourArea(contours[i])
        bounding_box_area = w * h

        if abs(contour_area - bounding_box_area) / bounding_box_area < 0.5:  # Ajustez la tolérance ici
            if bounding_box_area < 50000 and bounding_box_area > 800:  # Dégage les aires trop extrêmes
                if 2 * w > h:  # Car ce sont des lignes

                    finded = False
                    banned_liste = []

                    for i in range(len(dialogue_line)) :
                        x1, y1, x2, y2 = dialogue_line[i]["bbox"]
                        if min(abs(x - x2),abs(x-x1),abs(x+w-x1),abs(x+w-x2)) < 50 and abs(y - y1) < 20:
                            if finded == True and dialogue_line[i] not in banned_liste:
                                x3, y3, x4, y4 = dialogue_line[i-1]["bbox"]
                                dialogue_line[i]["bbox"] = ((min(x, x1,x3), min(y, y1,y3), max(x + w, x2,x4), max(y + h, y2,y4)))
                                banned_liste.append(i-1)

                            else:
                                dialogue_line[i]["bbox"] = ((min(x, x1), min(y, y1), max(x + w, x2), max(y + h, y2)))

                            finded = True

                    dialogue_line = [dialogue_line[i] for i in range(len(dialogue_line)) if i not in banned_liste] #On degage doublons

                    if finded == False: #Si on en trouve aucun on cree
                        dialogue_line.append({"bbox":(x, y, x + w, y + h)})

    #Verification
    f_lines = []
    for line in dialogue_line:
        x1, y1, x2, y2 = line["bbox"]
        if (x2-x1)*(y2-y1) > 1500 and (x2-x1) >= (y2-y1):
            f_lines.append(line)


    return f_lines


def group_into_bubble(dialogue_line):
    dialogue_bubble = [] #Liste des differentes bulles

    for line in dialogue_line:
        finded = False
        banned_liste = []

        left = line["bbox"][0]
        top = line["bbox"][1]
        right = line["bbox"][2]
        down = line["bbox"][3]

        for i in range(len(dialogue_bubble)) :
            x1, y1, x2, y2 = dialogue_bubble[i]["bbox"] #Passé

            if abs(y2 - top) < 40 and (x1 <= right and x2 >= left) :
                if finded == True and dialogue_bubble[i] not in banned_liste:
                    x3, y3, x4, y4 = dialogue_bubble[i-1]["bbox"]
                    dialogue_bubble[i]["bbox"] = ((min(left, x1,x3), min(top, y1,y3), max(right, x2,x4), max(down, y2,y4)))
                    banned_liste.append(i-1)
                    dialogue_bubble[i]["nb_line"] += 1

                else :
                    dialogue_bubble[i]["bbox"] = (min(left, x1), min(top, y1), max(right, x2), max(down, y2))
                    dialogue_bubble[i]["nb_line"] += 1

                finded = True

        dialogue_bubble = [dialogue_bubble[i] for i in range(len(dialogue_bubble)) if i not in banned_liste] #On degage les doublons

        if finded == False: #Si on en trouve aucun on cree
            line["nb_line"] = 1
            dialogue_bubble.append(line)


    #Extraction texte
    for bubble in dialogue_bubble:
        x1, y1, x2, y2 = bubble["bbox"]

        text_zone = image[y1:y2, x1:x2]

        hauteur, largeur, _ = text_zone.shape
        epaisseur_contour = int(0.1 * min(largeur, hauteur))
        image_contour = np.ones((hauteur + 2 * epaisseur_contour, largeur + 2 * epaisseur_contour, 3), dtype=np.uint8) * 255
        image_contour[epaisseur_contour:epaisseur_contour + hauteur, epaisseur_contour:epaisseur_contour + largeur, :] = text_zone

        # cv2.imwrite("temp.png", image_contour)
        # image_temp = cv2.imread("temp.png")

        image_temp = image_contour

        extracted_text = pytesseract.image_to_string(image_temp,lang = langue, config=config)
        print(extracted_text)

        if check_bubble(extracted_text) == True:
            texte_traite = pre_clean(extracted_text)
            bubble["text"] = texte_traite.strip()
        else :
            bubble["text"] = ""

    return dialogue_bubble


#Traduction
def traduction(text):
    text_traduit = str(translator.translate_text(text, target_lang="FR"))
    return text_traduit

#Adaptation du texte
def ini_h_ligne(zone):
    h_zone = abs(zone[0][1]-zone[1][1]) #hauteur de la zone
    h_ligne = h_zone/(zone[2]+inter_espace*(zone[2]-1)) #hauteur d'une ligne (1/4 ligne entre chaque = 0.25)

    return h_ligne


def longest_mot(mots):
    longest = [mots[0], cv2.getTextSize(mots[0], font, font_scale, font_thickness)[0]] #[mot le plus long, sa longeur]
    for mot in mots:
        longeur_mot = cv2.getTextSize(mot, font, font_scale, font_thickness)[0]
        if longeur_mot > longest[1] :
            longest = [mot,longeur_mot]

    return longest

#Initialisation taile police
def taille_police(font_scale,zone,text): #font_scale = taille de base (volontairement trop grosse)
    z_largeur = abs(zone[0][0] - zone[1][0]) #largeur de la zone
    mots = text_trad.split() #Sequencage en une liste des mots
    longest = longest_mot(mots)

    space_largeur = cv2.getTextSize(' ', font, font_scale, font_thickness)[0][0]
    longest_largeur = cv2.getTextSize(longest[0], font, font_scale, font_thickness)[0][0]
    longest_hauteur = cv2.getTextSize(longest[0], font, font_scale, font_thickness)[0][1]

    if (longest_largeur <= z_largeur) and (longest_hauteur <= h_ligne): #le mot le plus grand rentre
        if cv2.getTextSize(text, font, font_scale, font_thickness)[0][0] <= zone[2]*z_largeur :

            c_ligne = 1 #ligne actuelle
            c_place = z_largeur #places restante
            for mot in mots :
                longeur_mot = cv2.getTextSize(mot, font, font_scale, font_thickness)[0][0]
                if c_place - longeur_mot < 0 : #Le mot ne rentre pas dans la ligne

                    if c_ligne < zone[2]: #On saute une ligne
                        c_ligne += 1
                        c_place = z_largeur - longeur_mot - space_largeur

                    else : #Ca rentrera pas
                        return taille_police(font_scale-0.2,zone,text)


                else : #Ca rentre
                    c_place = c_place - longeur_mot - space_largeur #On enleve aussi l'espace
        else:
            return taille_police(font_scale-0.2,zone,text)

    else :
        return taille_police(font_scale-0.2,zone,text)

    return font_scale


#Insertion
def insert_text(font_scale,zone,text):
    font_scale = taille_police(font_scale,zone,text)
    space_largeur = cv2.getTextSize(' ', font, font_scale, font_thickness)[0][0]
    z_largeur = abs(zone[0][0] - zone[1][0]) #largeur de la zone
    mots = text_trad.split() #Sequencage en une liste des mots
    c_ligne = 1 #ligne actuelle
    c_place = z_largeur #places restante

    for mot in mots :
        longeur_mot = cv2.getTextSize(mot, font, font_scale, font_thickness)[0][0]
        if c_place - longeur_mot < 0 : #Le mot ne rentre pas dans la ligne
            c_ligne += 1
            c_place = z_largeur - longeur_mot - space_largeur

        else : #Ca rentre
            c_place = c_place - longeur_mot - space_largeur #On enleve aussi l'espace

        x_mot = int(zone[0][0] + z_largeur - c_place - longeur_mot - space_largeur)
        y_mot = int(zone[0][1] + inter_espace * h_ligne * (c_ligne-1) + c_ligne * h_ligne)

        u_font_thickness = int(2*(font_scale+0.6)) #Valeur d'ecriture

        cv2.putText(image, mot, (x_mot, y_mot), font, font_scale, color, u_font_thickness) # Écrit le texte sur l'image

#Remplacement de caractéres apres traduction, pour qu'ils puissent etre correctement ecrit
def replace_characters(text):
    text = text.upper() #On remet le texte en majuscule

    text = text.replace('É', 'E')
    text = text.replace('È', 'E')
    text = text.replace('Ê', 'E')
    text = text.replace('À', 'A')
    text = text.replace('Â', 'A')
    text = text.replace('Û', 'U')
    text = text.replace('Ù', 'U')
    text = text.replace('Î', 'I')
    text = text.replace('Ï', 'I')
    text = text.replace('ŒU', 'OEU')
    text = text.replace('Œ', 'OE')
    text = text.replace('Ô', 'O')
    text = text.replace('Ç', 'C')
    text = text.replace('’', "'")

    return text

#Verifie que le fichier est une image
def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.img', '.webp']

    # Vérifie si le fichier existe
    if not os.path.exists(file_path):
        return False

    # Vérifie si le fichier a une extension d'image prise en charge
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in image_extensions:
        return True
    else:
        return False


# Enregistrer l'image
def remove_file_extension(file_name):
    # Utiliser os.path.splitext pour séparer le nom de fichier et l'extension
    root, extension = os.path.splitext(os.path.basename(file_name))
    return root


# Utilisation des fonctions
directory_path_E = 'TradE/' # Remplacez par le chemin d'acces du dossier d'entree
directory_path_S = 'TradS/' # Remplacez par le chemin d'acces du dossier de sortie

fichiers = os.listdir(directory_path_E)
for fichier in fichiers:
    image_path = os.path.join(directory_path_E, fichier)

    if is_image_file(image_path):
        image = cv2.imread(image_path)

        contours = treat_image(image)
        dialogue_lines = group_text_into_line(contours)
        dialogue_bubbles = group_into_bubble(dialogue_lines)

        for bubble in dialogue_bubbles:
            if len(bubble['text']) > 0:
                text_trad = traduction(bubble['text'])
                if text_trad != bubble['text']: #Necessité de changement
                    text_trad = replace_characters(text_trad)

                    x1, y1, x2, y2 = bubble["bbox"]
                    background_color = median_color(bubble["bbox"],image)
                    cv2.rectangle(image, (x1, y1), (x2, y2), background_color, thickness=cv2.FILLED)
                    color = complementary_color(background_color) #Couleur pour ecrire


                    nb_lignes = bubble['nb_line']
                    zone = [(x1,y1),(x2,y2),nb_lignes]
                    h_ligne = ini_h_ligne(zone)

                    font_scale = 10 #Parametre de base volotairement élevé
                    insert_text(font_scale,zone, text_trad)


        nom_fichier = remove_file_extension(image_path) + "_trad.jpg"
        cv2.imwrite(directory_path_S + nom_fichier, image)

