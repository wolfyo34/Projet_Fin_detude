from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
import csv
from selenium.webdriver.common.keys import Keys
from pandas import DataFrame


def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://www.careerbuilder.com/jobs?posted=&pay=20&radius=30&emp=&cb_apply=false&keywords=Computer+&location=Dallas&cb_workhome=false")
driver.set_window_size(250, 720)
action = ActionChains(driver)

recommended_skills = []
description_tab = []
nom_entreprise_tab=[]
url_tab = []
titre_tab=[]
region='Arizona'
k=190


time.sleep(5)
for i in range(1, 25):
    path_annonce_1 = '/html/body/div[2]/div/div[1]/main/div/div[2]/div/div/div[1]/div/div[2]/div[3]/div[3]/div/ol/li['
    path_annonce_2 = ']/a[1]'
    path_annonce = path_annonce_1 + str(i) + path_annonce_2
    driver.find_element_by_xpath(path_annonce).click()
    time.sleep(2)
    url = driver.current_url
    url_tab.append(url)
    time.sleep(2)
    path_nom_entreprise = "data-details"
    time.sleep(2)
    nom_entreprise = driver.find_element(By.CLASS_NAME, "data-display-header_content")
    nom_entreprise_tab.append(nom_entreprise.text.split("\n")[1])
    titre_tab.append(nom_entreprise.text.split("\n")[0])
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, 2500)")
    time.sleep(2)
    
    path_description = "jdp_description"


    description = driver.find_element_by_id(path_description)
    time.sleep(2)
    description_tab.append(description.text)

    j = 0
    desc = description_tab[i-1].split("\n")
    a=0
    
    while j <(len(desc)):
        
        if "Help us improve CareerBuilder" in desc[j]:
            a=2
        
        if a==1:
            le_split = desc[j].split(" ")
            recommended_skills.append(le_split)
        
        if "Recommended Skills" in desc[j]:
            a=1
            
        j+=1
    k+=1
    time.sleep(2)
    reg=nom_entreprise_tab[i-1].split(",")
    reg=reg[1][1:3]
    beta_code="01	Alabama	Ala.	AL 02	Alaska		AK 04	Arizona	Ariz.	AZ 05	Arkansas	Ark.	AR 06	Californie	Calif.	CA 08	Colorado	Colo.	CO 09	Connecticut	Conn.	CT 10	Delaware	Del.	DE 11	District de Columbia	D.C.	DC 12	Floride	Fla.	FL 13	Géorgie	Ga.	GA 15	Hawaii		HI 16	Idaho		ID 17	Illinois	Ill.	IL 18	Indiana	Ind.	IN 19	Iowa		IA 20	Kansas	Kans.	KS 21	Kentucky	Ky.	KY 22	Louisiane		LA 23	Maine	Me.	ME 24	Maryland	Md.	MD 25	Massachusetts	Mass.	MA 26	Michigan	Mich.	MI 27	Minnesota	Minn.	MN 28	Mississippi	Miss.	MS 29	Missouri	Mo.	MO 30	Montana	Mont.	MT 31	Nebraska	Nebr.	NE 32	Nevada	Nev.	NV 33	New Hampshire	N.H.	NH 34	New Jersey	N.J.	NJ 35	Nouveau-Mexique	N.Mex.	NM 36	New York	N.Y.	NY 37	Caroline du Nord	N.C.	NC 38	Dakota du Nord		ND 39	Ohio		OH 40	Oklahoma	Okla.	OK 41	Oregon	Ore.	OR 42	Pennsylvanie	Penn.	PA 44	Rhode Island	R.I.	RI 45	Caroline du Sud	S.C.	SC 46	Dakota du Sud		SD 47	Tennessee	Tenn.	TN 48	Texas	Tex.	TX 49	Utah		UT 50	Vermont	Vt.	VT 51	Virginie	Va.	VA 53	Washington	Wash.	WA 54	Virginie-Occidentale	W.Va.	WV 55	Wisconsin	Wis.	WI 56	Wyoming	Wyo.	WY"
    alpha_code=['AL','AK','AZ','AR','CA','CO','CT','DE','DC','GA','FL','HI','ID','IN','IA','IL','KS','KL','LA','ME','MD','MA','MI','MS','MO','MT','MN','NE','NV','NH','NJ','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','MY']
    beta_code=beta_code.replace("\t"," ")
    ceta_code=beta_code.split(" ")
    for e in range(len(ceta_code)):
        if reg==ceta_code[e]:
            if not ceta_code[e-3].isdigit():
                if not ceta_code[e-4].isdigit():
                    region=ceta_code[e-4]+" "+ceta_code[e-3]+" "+ceta_code[e-2]
                    break
                region=ceta_code[e-3]+" "+ceta_code[e-2]
                break
            region=ceta_code[e-2]
            break
    C = {'id': k,'nom_entreprise': [nom_entreprise_tab[i-1]], 'titre': [titre_tab[i-1]],'region':region,'salaire':nom_entreprise.text.split("\n")[2] ,'description': [description_tab[i - 1]], 'competence': [recommended_skills], 'url':[url_tab[i-1]]}
    donnees = DataFrame(C, columns=['id', 'nom_entreprise', 'titre','region','salaire','description', 'competence', 'url'])
    export_csv = donnees.to_csv("jobs5.csv", mode='a', index=None, header=False, encoding='utf8', sep=';')
    time.sleep(1)
    driver.execute_script("window.history.go(-1)")
    driver.execute_script("window.scrollTo(0, 800)")
    recommended_skills =[]
    i += 1
    
    time.sleep(3)