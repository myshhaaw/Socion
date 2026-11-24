"""Synthetic civil-unrest news dataset generator (text is synthetic; places + regional context come from the ACLED file)."""
import pandas as pd, numpy as np, random

SEED = 42
rng = random.Random(SEED); np.random.seed(SEED)
from pathlib import Path
ACLED = str(Path(__file__).resolve().parents[2]/"data/raw/acled_asia_pacific_weekly.xlsx")
N_POS, N_NEG = 1015, 435

# ---------- real geography + regional context from ACLED ----------
a = pd.read_excel(ACLED)
countries = ['India','Pakistan','Bangladesh','Sri Lanka','Nepal','Philippines','Indonesia','Thailand']
cw = {'India':5,'Pakistan':2,'Bangladesh':2,'Sri Lanka':1,'Nepal':1.5,'Philippines':1,'Indonesia':1.5,'Thailand':1}
pr = a[a.COUNTRY.isin(countries) & a.EVENT_TYPE.isin(['Protests','Riots'])]
tot = pr.groupby(['COUNTRY','ADMIN1']).EVENTS.sum().reset_index()
states = {c: tot[tot.COUNTRY==c].nlargest(8,'EVENTS') for c in countries}
weekly = pr.groupby(['COUNTRY','ADMIN1','WEEK']).EVENTS.sum()
all_weeks = pd.date_range('2016-12-31','2023-12-30',freq='7D')
series = {}
def get_series(c, s):
    if (c,s) not in series:
        try: ser = weekly.loc[(c,s)].reindex(all_weeks, fill_value=0)
        except KeyError: ser = pd.Series(0, index=all_weeks)
        series[(c,s)] = ser
    return series[(c,s)]
def context(c, s, d):
    ser = get_series(c, s)
    idx = ser.index.searchsorted(pd.Timestamp(d), side='right') - 1  # week containing d
    prev4 = int(ser.iloc[max(0, idx-4):idx].sum())
    base = ser.iloc[max(0, idx-52):idx].mean() * 4
    ratio = round(prev4 / base, 2) if base > 0 else 1.0
    return prev4, min(ratio, 5.0)

def pick_place():
    c = rng.choices(countries, weights=[cw[x] for x in countries])[0]
    t = states[c]; s = rng.choices(t.ADMIN1.tolist(), weights=np.sqrt(t.EVENTS).tolist())[0]
    return c, s
def rand_date():
    return pd.Timestamp('2018-01-01') + pd.Timedelta(days=rng.randint(0, 2190))

# ---------- vocab ----------
CAUSES = {
 'prices_cost_of_living': [("the fuel price hike","a rollback of the fuel price hike"),("rising food prices","relief from rising food prices"),("the cooking gas price increase","withdrawal of the cooking gas price increase"),("soaring vegetable prices","price controls on soaring vegetable prices"),("the electricity tariff hike","a cut to the electricity tariff hike")],
 'wages_labour': [("unpaid wages","immediate payment of unpaid wages"),("delayed salaries","release of delayed salaries"),("the lack of job regularisation","regularisation of contract jobs"),("pending pension dues","clearance of pending pension dues"),("a low minimum wage","a higher minimum wage"),("unsafe working conditions","safer working conditions")],
 'land_environment': [("land acquired for a highway project","fair compensation for land acquired for a highway project"),("a proposed dam","cancellation of a proposed dam"),("a polluting factory","closure of a polluting factory"),("illegal sand mining","a ban on illegal sand mining"),("tree felling in a forest area","protection of a forest area from tree felling")],
 'education_exams': [("an exam paper leak","a fresh exam after the paper leak"),("the tuition fee hike","a rollback of the tuition fee hike"),("delayed recruitment results","early release of delayed recruitment results"),("delayed scholarships","release of delayed scholarships")],
 'public_services': [("the acute water shortage","an end to the acute water shortage"),("long power cuts","an end to long power cuts"),("broken roads","repair of broken roads"),("the shortage of doctors at the local hospital","more doctors at the local hospital"),("irregular garbage collection","regular garbage collection")],
 'governance_justice': [("the arrest of a local leader","release of a local leader arrested last week"),("alleged corruption by local officials","action against alleged corruption by local officials"),("a custodial death","a judicial inquiry into a custodial death"),("the transfer of a popular district official","reversal of the transfer of a popular district official"),("police inaction in a murder case","arrest of suspects in a murder case")],
 'political_electoral': [("alleged vote rigging","a recount over alleged vote rigging"),("the disputed election result","a review of the disputed election result"),("the rejection of nomination papers","reinstatement of rejected nomination papers")],
 'housing_eviction': [("an eviction drive","a halt to an eviction drive"),("the demolition of homes","rehabilitation after the demolition of homes"),("the removal of street vendors","licences after the removal of street vendors")],
}
PARTS = {
 'prices_cost_of_living': ["residents","traders","commuters","housewives","transport operators","opposition supporters","shopkeepers"],
 'wages_labour': ["garment workers","sanitation workers","teachers","nurses","factory workers","plantation workers","bus drivers","contract employees","port workers","power utility staff"],
 'land_environment': ["villagers","farmers","fishermen","tribal communities","local residents","environmental activists"],
 'education_exams': ["students","university students","job aspirants","parents and students","graduates"],
 'public_services': ["residents","villagers","slum dwellers","local women","shopkeepers","locals"],
 'governance_justice': ["supporters","lawyers","local residents","activists","family members and neighbours","students"],
 'political_electoral': ["party supporters","opposition workers","candidates' supporters","local voters"],
 'housing_eviction': ["slum residents","street vendors","families","residents","shopkeepers"],
}
TYPE_CAT_W = {  # which causes suit which event type
 'protest': {'prices_cost_of_living':2,'wages_labour':2,'land_environment':2,'education_exams':2,'public_services':3,'governance_justice':2,'political_electoral':1,'housing_eviction':1.5},
 'riot':    {'prices_cost_of_living':2,'governance_justice':3,'political_electoral':3,'housing_eviction':2,'public_services':1,'education_exams':1},
 'strike':  {'wages_labour':8,'prices_cost_of_living':1,'public_services':1,'governance_justice':1},
 'march':   {'prices_cost_of_living':2,'wages_labour':2,'land_environment':2,'education_exams':2,'governance_justice':2,'political_electoral':1,'public_services':1},
 'blockade':{'public_services':3,'land_environment':2,'prices_cost_of_living':2,'governance_justice':2,'housing_eviction':1,'wages_labour':1},
}
TYPE_W = {'protest':38,'riot':14,'strike':18,'march':14,'blockade':16}
TRIG = {
 'protest': {'past':["staged a protest","held a demonstration","staged a sit-in","gathered outside the district office","protested"],
             'ongoing':["are protesting","continued their sit-in","are staging a sit-in","are demonstrating"],
             'planned':["plan to stage a protest","have announced a protest","will hold a demonstration","are set to protest"]},
 'riot':   {'past':["clashed with police","pelted stones at security forces","set vehicles on fire","ransacked shops","torched a government vehicle"],
             'ongoing':["are clashing with police","are pelting stones at security forces","are setting tyres on fire"]},
 'strike': {'past':["went on strike","downed tools","began an indefinite strike","launched a strike"],
             'ongoing':["are on strike","continued their strike","remain on strike"],
             'planned':["will go on strike","have called a strike","plan a two-day strike","plan to go on strike"]},
 'march':  {'past':["marched to the state assembly","took out a rally","held a long march to the capital","marched through the city"],
             'ongoing':["are marching towards the capital","continued their march","are marching to the secretariat"],
             'planned':["plan a march to the state assembly","will take out a rally","have announced a march to the capital"]},
 'blockade':{'past':["blocked the national highway","blockaded the main road","blocked rail traffic","blocked the road to the district office"],
             'ongoing':["are blocking the national highway","continued their road blockade","are blockading the main road"],
             'planned':["plan to block the national highway","have threatened a road blockade","will block rail traffic"]},
}
STATUS_W = {'protest':{'past':60,'ongoing':20,'planned':20},'riot':{'past':70,'ongoing':30},'strike':{'past':35,'ongoing':30,'planned':35},
            'march':{'past':45,'ongoing':25,'planned':30},'blockade':{'past':55,'ongoing':25,'planned':20}}
CROWD = {'small':["about 40","a few dozen","around 60"],'medium':["hundreds of","around 300","nearly 500"],'large':["thousands of","over 2,000","more than 5,000"]}
POLICE = {'water cannon':"Police used water cannon to disperse the crowd.",'baton charge':"Police resorted to a baton charge.",
          'tear gas':"Police fired tear gas shells to disperse the crowd.",'rubber bullets':"Security forces fired rubber bullets.",
          'live fire':"Security forces opened fire.",'none':"The situation remained peaceful."}
POLICE_SHORT = {'water cannon':"water cannon used",'baton charge':"baton charge",'tear gas':"tear gas fired",'rubber bullets':"rubber bullets fired",'live fire':"shots fired",'none':""}
OUTCOME = {'past':["Officials said talks with the representatives would continue.","The local administration said it is monitoring the situation.","Traffic in the area was disrupted for several hours.","Police said a case has been registered.","Organisers said the agitation would continue until their demands are met."],
           'ongoing':["Officials have invited the representatives for talks.","Normal life remains disrupted in the area.","Organisers vowed to continue until their demands are met.","The administration has deployed additional personnel."],
           'planned':["Organisers said they expect a large turnout.","The administration has not yet granted permission.","Officials said security arrangements are being reviewed.","Leaders said the decision was taken after talks failed."]}
ORD = ["second","third","fourth","fifth","sixth"]
OPENERS = ["BREAKING:","Update:","Happening now:","Live:","Ground report:","Just in:","Reports coming in:",""]
TYPE_TAG = {'protest':"#Protest",'riot':"#Riot",'strike':"#Strike",'march':"#Rally",'blockade':"#RoadBlock"}
GEN_TAGS = ["#News","#Breaking","#Update","#Ground","#LocalNews"]

def cap(s): return s[0].upper()+s[1:]
def hashstate(s): return "#"+s.replace(" ","").replace("-","")

# ---------- positive events ----------
def make_event():
    c, s = pick_place()
    et = rng.choices(list(TYPE_W), weights=list(TYPE_W.values()))[0]
    st = rng.choices(list(STATUS_W[et]), weights=list(STATUS_W[et].values()))[0]
    cat = rng.choices(list(TYPE_CAT_W[et]), weights=list(TYPE_CAT_W[et].values()))[0]
    cause = rng.choice(CAUSES[cat]); part = rng.choice(PARTS[cat])
    crowd = rng.choices(['small','medium','large'], weights=[30,45,25])[0]
    if st == 'planned': police, inj, fat = 'none', 0, 0
    else:
        pw = {'riot':[10,25,28,17,10,10],'protest':[5,15,12,3,1,64],'strike':[1,3,2,0,0,94],'march':[4,10,8,1,0,77],'blockade':[4,15,10,2,1,68]}[et]
        police = rng.choices(list(POLICE), weights=pw)[0]
        pinj = {'riot':0.75,'protest':0.15,'strike':0.05,'march':0.12,'blockade':0.18}[et]
        if police in ('rubber bullets','live fire'): pinj = 0.8
        inj = rng.choice([1,2,3,5,8,12,20,35]) if rng.random() < pinj else 0
        pf = {'riot':0.30,'protest':0.03,'strike':0.01,'march':0.02,'blockade':0.05}[et]
        if police == 'live fire': pf = 0.5
        fat = rng.choice([1,1,1,2,3,5]) if rng.random() < pf else 0
    date = rand_date()
    return dict(c=c,s=s,et=et,st=st,cat=cat,cause=cause,part=part,crowd=crowd,police=police,inj=inj,fat=fat,date=date)

def risk_score(ev, prev4, ratio):
    sc = {'riot':1.1,'blockade':0.5,'protest':0.0,'strike':0.0,'march':-0.4}[ev['et']]
    sc += {'live fire':2.0,'rubber bullets':1.4,'tear gas':1.0,'baton charge':0.9,'water cannon':0.6,'none':0}[ev['police']]
    sc += 0.8*(ev['inj']>0) + 2.0*(ev['fat']>0) + {'small':0,'medium':0.4,'large':1.0}[ev['crowd']]
    sc += 0.7*(ratio>1.3) + 0.5*(ratio>1.8) + 0.4*(prev4>=40)
    sc -= 0.5*(ev['st']=='planned')
    return sc + np.random.normal(0, 0.8)

def subj(ev, show_crowd):
    if show_crowd and ev['st'] != 'planned': return rng.choice(CROWD[ev['crowd']]) + " " + ev['part']
    return ev['part']
def timephrase(ev, art_date, d_event):
    if ev['st']=='past': return "on "+d_event.strftime('%A')
    if ev['st']=='ongoing': return "for the "+rng.choice(ORD)+" day"
    return rng.choice(["next week","on "+(art_date+pd.Timedelta(days=rng.randint(2,5))).strftime('%A'),"later this week"])

def harm_bits(ev, p_state):
    bits = {}
    if ev['st'] != 'planned':
        if rng.random() < p_state: bits['police'] = ev['police']
        if rng.random() < p_state and ev['inj']>0: bits['inj'] = ev['inj']
        if rng.random() < p_state and ev['fat']>0: bits['fat'] = ev['fat']
    return bits

def write_article(ev, ttype, art_date):
    pstate = {'headline':0.35,'social_post':0.4,'short_article':0.9}[ttype]
    show_crowd = rng.random() < {'headline':0.35,'social_post':0.45,'short_article':0.85}[ttype]
    show_cause = rng.random() < {'headline':0.88,'social_post':0.85,'short_article':0.97}[ttype]
    trig = rng.choice(TRIG[ev['et']][ev['st']])
    P = subj(ev, show_crowd); S = ev['s']; tp = timephrase(ev, art_date, ev['date'])
    c, d = ev['cause']; bits = harm_bits(ev, pstate)
    over = rng.choice(["over","against","amid anger over"]) if ev['et']!='strike' else rng.choice(["over","against"])
    use_d = rng.random() < 0.3
    cause_clause = ""
    if show_cause: cause_clause = f"demanding {d}" if use_d else f"{over} {c}"
    inj_txt = f"{bits['inj']} {'person was' if bits['inj']==1 else 'people were'} injured." if 'inj' in bits else ""
    fat_txt = ("One person was killed." if bits['fat']==1 else f"{bits['fat']} people were killed.") if 'fat' in bits else ""
    pol_txt = POLICE[bits['police']] if 'police' in bits else ""
    if ttype == 'headline':
        t = rng.choice([f"{S}: {cap(P)} {trig} {cause_clause}".strip(),
                        f"{cap(P)} {trig} in {S} {tp} {cause_clause}".strip(),
                        f"{cap(P)} {trig} {cause_clause} in {S}".strip()])
        tail = []
        if 'fat' in bits: tail.append(f"{bits['fat']} dead")
        elif 'inj' in bits: tail.append(f"{bits['inj']} injured")
        if 'police' in bits and POLICE_SHORT[bits['police']]: tail.append(POLICE_SHORT[bits['police']])
        if tail: t += ", " + ", ".join(tail)
    elif ttype == 'social_post':
        op = rng.choice(OPENERS); tags = []
        if rng.random()<0.6: tags.append(hashstate(S))
        if rng.random()<0.5: tags.append(TYPE_TAG[ev['et']])
        if rng.random()<0.4: tags.append(rng.choice(GEN_TAGS))
        body = rng.choice([f"{cap(P)} {trig} in {S} {tp} {cause_clause}.",
                           f"{cap(P)} {trig} in {S}! {cause_clause.capitalize() if cause_clause else ''}".strip()+("." if cause_clause else "")])
        extra = rng.choice([inj_txt, fat_txt or inj_txt, pol_txt]) if (inj_txt or fat_txt or pol_txt) else ""
        t = " ".join(x for x in [op, body, extra, " ".join(tags)] if x).replace("  "," ")
    else:
        s1 = f"{S}: {cap(P)} {trig} {tp} {cause_clause}.".replace(" .", ".").replace("  "," ")
        s2 = " ".join(x for x in [pol_txt, fat_txt or inj_txt] if x)
        if fat_txt and inj_txt: s2 = " ".join([pol_txt, fat_txt, inj_txt]).strip()
        s3 = rng.choice(OUTCOME[ev['st']]) if rng.random()<0.8 else ""
        t = " ".join(x for x in [s1, s2, s3] if x)
    t = " ".join(t.split()).replace(" ,", ",").replace(" .", ".").replace(" !", "!").strip()
    cause_label = d if use_d else c   # label = the exact phrase stated in the text
    assert trig in t and S in t and ((cause_label in t) or not show_cause)
    return dict(text=t, trigger=trig,
                cause=cause_label if show_cause else "not stated", cause_cat=ev['cat'] if show_cause else "not stated",
                crowd=ev['crowd'] if (show_crowd and ev['st']!='planned') else "not stated",
                police=bits.get('police','not stated'), inj=bits.get('inj', 0 if (ev['st']!='planned' and 'police' in bits) else np.nan),
                fat=bits.get('fat', 0 if (ev['st']!='planned' and ('inj' in bits or 'police' in bits)) else np.nan))

# ---------- hard negatives ----------
TEAMS = ["district","university","state league","women's","under-19","veterans'"]
NEG = {
 'sports_or_weather': ["The {S} opener's blistering strike rate set up a six-wicket win in the {t} tournament.","Strikers from {S} FC ended a long goal drought with a late winner on Sunday.","A lightning strike killed two cattle as heavy rain lashed {S}.","The {S} bowler's final-over strike turned the match in the {t} final.","Cyclone warning issued for coastal {S}; fishermen advised to stay ashore.","{S} team strike a deal with a new kit sponsor ahead of the season."],
 'political_campaign': ["Ruling party leader addresses election rally in {S}, promises a new jobs scheme.","Roadshow draws large crowds as campaign enters final week in {S}.","Party workers distribute pamphlets ahead of local body polls in {S}.","Opposition candidate files nomination in {S} amid cheering supporters.","Campaign rally in {S} focuses on farm loans and electricity, leaders say."],
 'diplomatic_protest': ["{C} lodges a strong protest with a foreign mission over delayed visas.","Foreign ministry hands over a protest note after a fishing boat was seized.","{C} registers a diplomatic protest over remarks by a visiting official."],
 'figurative_use': ["Analysts say low turnout in {S} was a protest vote against both major parties.","Critics call the new tax a quiet protest against rising costs, but no gatherings were held.","The film's director said the song is a protest against modern life; it released in {S} on Friday."],
 'historical_reference': ["New book revisits the student uprising of {y} in {C}.","Museum in {S} opens an exhibition on the history of the labour movement.","Documentary on the historic farmers' march premieres in {S}.","Historians in {S} debate the legacy of the {y} movement at a seminar."],
 'opinion_analysis': ["Opinion: why protests rarely change policy in {C}.","Analysis: what rising fuel prices mean for household budgets in {S}.","Experts warn unemployment could trigger unrest in {C}, but no demonstrations have been reported so far.","Column: lessons from past strikes for the {sector} sector in {C}."],
 'policy_announcement': ["State government announces a wage hike for contract workers in {S}.","Cabinet approves a tuition fee cap for colleges in {S}.","Power utility says it will end load-shedding in {S} within two weeks.","Authorities in {S} unveil a new water supply scheme for {n} villages.","Government clears pending pension dues for retired teachers in {S}."],
 'cultural_event': ["Thousands join the annual harvest festival procession in {S}.","Over {n} runners take part in the {S} city marathon on Sunday.","National day parade held in {S} under tight but routine security.","Crowds gather for the opening of a new bridge in {S}."],
 'routine_news': ["Heavy rain disrupts road traffic in {S}; officials ask residents to stay indoors.","Highway reopened after a landslide blocked traffic for hours in {S}.","Unemployment in {S} fell to {p}% last quarter, a report says.","Power outage hits {n} households in {S} after a transformer fault.","Wedding season boosts sales for traders in {S}, say market associations.","Exam results for {sector} diploma courses declared in {S}."],
}
SECT = ["textile","transport","banking","power","education","health"]
def make_negative(i):
    c, s = pick_place(); date = rand_date()
    nt = rng.choices(list(NEG), weights=[12,12,8,8,10,12,12,12,12])[0]
    t = rng.choice(NEG[nt]).format(S=s, C=c, t=rng.choice(TEAMS), y=rng.randint(1985,2015), n=rng.choice([40,120,300,800,2000,5000]), p=round(rng.uniform(3,9),1), sector=rng.choice(SECT))
    ttype = rng.choices(['headline','social_post','short_article'], weights=[45,30,25])[0]
    if ttype == 'social_post':
        t = " ".join(x for x in [rng.choice(OPENERS), t, hashstate(s), rng.choice(["#News","#Update","#Local",""])] if x)
    elif ttype == 'short_article':
        t = f"{s}: {t} " + rng.choice(["Officials said further details would be shared later.","The announcement was made on Monday.","Residents welcomed the development.","The matter drew little public attention.","Local media reported the development."])
    return dict(text=t.replace("  "," ").replace(" .",".").strip(), country=c, state=s, date=date, nt=nt, ttype=ttype)

# ---------- assemble ----------
rows, ev_id, art_id = [], 0, 0
SEEN = set()
pos_scores = []
events = []
n_pos = 0
while n_pos < N_POS:
    ev = make_event(); k = rng.choices([1,2,3], weights=[20,45,35])[0]; k = min(k, N_POS-n_pos)
    ev['prev4'], ev['ratio'] = context(ev['c'], ev['s'], ev['date'])
    ev['score'] = risk_score(ev, ev['prev4'], ev['ratio'])
    ev['k'] = k; ev['id'] = f"EV{ev_id:04d}"; ev_id += 1; n_pos += k; events.append(ev)
lo, hi = np.quantile([e['score'] for e in events], [0.45, 0.80])  # low 45% / medium 35% / high 20%
SRC = {'headline':['wire_style','regional_daily','tv_news'],'short_article':['wire_style','regional_daily','tv_news'],'social_post':['social_media']}
for ev in events:
    risk = 'low' if ev['score'] < lo else ('medium' if ev['score'] < hi else 'high')
    types = rng.sample(['headline','social_post','short_article'], ev['k']) if ev['k']<=3 else ['headline']
    for ttype in types:
        art_date = ev['date'] if ev['st']!='planned' else ev['date'] - pd.Timedelta(days=rng.randint(1,4))
        art_date = art_date + pd.Timedelta(days=rng.randint(0,2)) if ev['st']=='past' else art_date
        for _try in range(30):
            w = write_article(ev, ttype, art_date)
            if w['text'] not in SEEN: break
        SEEN.add(w['text'])
        tone = -0.15 - (0.3 if ev['et']=='riot' else 0) - (0.4 if ev['fat']>0 else 0) - (0.15 if ev['inj']>0 else 0) + (0.1 if ev['st']=='planned' else 0) + np.random.normal(0,0.12)
        rows.append(dict(event_id=ev['id'], date=art_date, source_type=rng.choice(SRC[ttype]), text_type=ttype, text=w['text'],
            country=ev['c'], state=ev['s'], is_unrest_event=1, negative_type="", event_type=ev['et'], event_status=ev['st'],
            trigger_phrase=w['trigger'], cause=w['cause'], cause_category=w['cause_cat'], crowd_size=w['crowd'],
            police_response=w['police'], injuries=w['inj'], fatalities=w['fat'], escalation_risk=risk,
            state_protest_riot_prev4w=ev['prev4'], state_trend_ratio=ev['ratio'], tone_score=round(float(np.clip(tone,-1,1)),2),
            split=('train' if ev['date'].year<=2021 else 'val' if ev['date'].year==2022 else 'test')))
for i in range(N_NEG):
    for _try in range(60):
        n = make_negative(i)
        if n['text'] not in SEEN: break
    SEEN.add(n['text']); prev4, ratio = context(n['country'], n['state'], n['date'])
    rows.append(dict(event_id="", date=n['date'], source_type=rng.choice(SRC[n['ttype']]), text_type=n['ttype'], text=n['text'],
        country=n['country'], state=n['state'], is_unrest_event=0, negative_type=n['nt'], event_type="none", event_status="none",
        trigger_phrase="", cause="", cause_category="", crowd_size="", police_response="", injuries=np.nan, fatalities=np.nan,
        escalation_risk="none", state_protest_riot_prev4w=prev4, state_trend_ratio=ratio,
        tone_score=round(float(np.clip(np.random.normal(0.1,0.25),-1,1)),2),
        split=('train' if n['date'].year<=2021 else 'val' if n['date'].year==2022 else 'test')))
df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
df.insert(0, 'article_id', [f"A{i:05d}" for i in range(len(df))])
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
df.to_csv(str(Path(__file__).resolve().parents[2]/"data/raw/synthetic_civil_unrest_news.csv"), index=False)
print(df.shape)
