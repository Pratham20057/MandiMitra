# MandiMantra Model Training — Hinglish Guide (Teacher/Viva ke liye)

Yeh guide project ke actual `train.py`, `ml_server.py` aur current `accuracy_report.csv` ke basis par hai. Iska purpose hai ki aap model training ko samjho bhi aur teacher ko apne words mein explain bhi kar sako.

Project Ghaziabad ke paanch crops ke mandi **Modal Price (₹/quintal)** ka 1, 7 aur 14 din aage forecast karta hai: Green Chilli, Potato, Rice, Tomato aur Wheat.

## 1. Teacher ko short mein kya bolna hai?

> Hum historical mandi modal prices ko crop ke hisaab se daily time series mein convert karte hain. Har date ke liye past prices se trend, momentum, volatility, recent range aur season ke features banate hain. LightGBM in features aur aage ke price change ke beech ka pattern seekhta hai. Hum model ko future dates par test karte hain jo training mein nahi thi, aur horizon jitna gap rakhte hain taaki target price training se test mein leak na ho. Phir model ko “price same rahega” aur moving-average jaise simple baselines se compare karte hain. Agar ML model baseline se better nahi hai, API baseline return karti hai. Isse bina evidence ke complex model ko best nahi bolte.

## 2. Model asal mein kya predict karta hai?

Model seedha rupee price nahi, balki **log return**, yaani price ka proportional change predict karta hai:

`target(t, h) = ln(price(t + h) / price(t))`

- `t` = aaj ki training row/date
- `h` = 1, 7, ya 14 din
- `price(t)` = aaj ka modal price
- `price(t+h)` = h din baad ka modal price
- `ln` = natural logarithm

Prediction ke waqt ise rupee price mein badalte hain:

`predicted price = current price × exp(predicted log return)`

**Simple example:** aaj ka price ₹1,000/quintal hai aur predicted log return 0.02 hai, to forecast `1000 × exp(0.02)`, lagbhag ₹1,020.20 hoga. Exponential conversion predicted price ko positive rakhta hai. Is implementation ka target `Modal_Price` hai; `Min_Price` aur `Max_Price` input file mein hain, lekin target nahi.

## 3. Kaggle data se model-ready data kaise banta hai?

Dataset mein State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min_Price, Max_Price aur Modal_Price jaise columns hain.

Preparation mein:

1. Crop names normalize hote hain, jaise `Chili`/`Chilli` ko `Green Chilli`, `Aloo` ko `Potato`, `Paddy` ko `Rice`.
2. State Uttar Pradesh aur District Ghaziabad par filter hota hai.
3. Sirf paanch supported crops aur valid date/positive modal price rakhe jaate hain.
4. Chuni hui market series ko filter kiya jata hai. Current preparation mein `Ghaziabad` aur `Ghaziabad APMC` naam ek `Ghaziabad` series mein merge kiye gaye hain.
5. Agar official Agmarknet feed mein kisi date/crop ka quote hai, same date/crop ka Kaggle quote usse replace hota hai.

**Market scope ka zaroori point:** Is setup ka matlab apne aap “Ghaziabad district ki har mandi ka combined forecast” nahi hota. Yeh prepared file mein chuni aur alias-merge ki gayi Ghaziabad market series hai. Agar target sabhi mandiyon ka combined district price ho, pehle un sab mandiyon ko date/crop ke hisaab se clearly aggregate karna hoga, phir dobara validation karni hogi.

## 4. Training ke har step ka reason

### Step 1: Date, crop, price read aur filter karna

Script CSV se date, commodity aur modal-price columns identify karti hai, date parse karti hai, invalid date/non-positive price hataati hai, requested district/market/date range lagati hai aur chronological order mein sort karti hai.

**Kyun?** Galat district, crop ya date ka record model ko galat market pattern sikhayega. Chronological order time-series forecast ke liye zaroori hai.

### Step 2: Har crop ki daily series banana

Ek hi din/crop ke multiple quotes ka average liya jata hai. 15-day rolling median ke comparison mein bahut zyada ya bahut kam quote—median se 2.5 guna upar ya 0.4 guna neeche—outlier mark hota hai. Daily calendar banakar 7 din tak ke chhote gaps interpolate hote hain; bade gaps fill nahi hote. Training ke liye crop mein kam-se-kam 500 usable days chahiye.

**Kyun?** Model ko regular date-index aur enough history chahiye. Interpolation chhote missing gaps ko handle karti hai; lekin interpolated number official mandi quote nahi hota. Missing quote ka matlab yeh nahi ki price waise hi raha.

### Step 3: Features banana — past ko useful signals mein badalna

Feature ka matlab input clue/signal hai jisse model pattern seekhta hai. Code mein:

- **Lagged returns (1, 2, 3, 5, 7, 10, 14, 21, 28, 56 din):** haal mein price upar/neeche ja raha hai ya nahi—momentum/reversal.
- **Rolling average se distance (7, 14, 30, 60, 90 din):** current price apne recent trend se kitna door hai.
- **Volatility (14, 30 din):** recent price movement kitna unstable hai.
- **30-day high/low aur range position:** current price recent range ke kis hisse mein hai.
- **Season/calendar features:** day-of-year ka sine/cosine, month aur weekday. Sine/cosine saal ke aakhir aur shuru ko seasonally paas dikhate hain.
- **One-year return:** lagbhag ek saal pehle ke mukable price change.

**Kyun?** Raw prices ki list se zyada, in signals se model ko recent direction, trend, risk/volatility aur seasonal timing samajhne mein madad milti hai.

### Step 4: Har crop aur horizon ka alag target

Har crop ke liye 1-day, 7-day aur 14-day future log return banaya jata hai. Isliye maximum 5 × 3 = **15 configurations/models** bante hain.

**Kyun alag?** Kal ka price aur 14 din baad ka price ek jaise problem nahi. Horizons badhne par uncertainty aur useful pattern badal sakta hai. Crops ke seasons aur price behavior bhi alag hain.

### Step 5: Past se train, future par test

Main evaluation mein history ka last 12 months test hota hai; history chhoti ho to 6 months. Pehle ke records training mein hote hain. Data kam padne par chronological 80/20 split fallback hai. Dates ko randomly shuffle nahi karte.

**Kyun random split nahi?** Random split future observations ko training mein aur purane observations ko test mein daal sakta hai. Real life mein future data pehle available nahi hota, isliye random split se score zaroorat se zyada achha aur misleading ho sakta hai.

**Embargo/gap kya hai?** Train aur test ke beech forecast horizon jitna gap. Misaal: 14-day target wali training row ka future target boundary ke paar test period mein na chala jaye. Isse label leakage ka risk kam hota hai.

Training ke andar LightGBM ke early stopping ke liye aakhri 15% chronological validation tail hoti hai aur wahan bhi horizon jitna gap rakha jata hai.

### Step 6: Recency weights

Har training row ko weight milta hai: `exp(-age_days / (365.25 × 3))`. Recent records ka weight zyada, purane ka dheere-dheere kam hota hai.

**Kyun?** Mandi behavior aur market conditions waqt ke saath badal sakte hain; isliye haal ki history zyada relevant ho sakti hai. Purane data ko poori tarah discard bhi nahi karte. “3 years” ek chosen decay scale hai, koi natural law nahi.

### Step 7: Baselines se fair comparison

Do simple forecasts benchmark bante hain:

1. **Persistence:** agle din/horizon ka price current price ke barabar.
2. **Moving average:** haal ke kam-se-kam 7 din (ya longer horizon) ka average.

Test period ka MAPE dekhkar best baseline choose hota hai. Agar LightGBM ka MAPE usse kam hai tabhi metadata mein ML select hota hai. Warna API baseline chalaati hai. Isliye screen par same price aaye to mumkin hai system ne jaan-boojhkar “price same rahega” baseline chuna ho.

## 5. LightGBM — detail mein, aasan Hinglish

### LightGBM ka full form aur category

**LightGBM = Light Gradient Boosting Machine.** Yeh Microsoft ka gradient-boosted decision-tree framework hai. `LGBMRegressor` regression use case hai, kyunki output continuous number hai—yahan price ka log return.

- **Decision tree:** sawal-based rules ki tarah data ko parts mein baantne wala model. Jaise: “volatility zyada hai?”, “price 30-day average se upar hai?”
- **Ensemble:** kai chhote models/trees milkar ek final prediction dete hain.
- **Boosting:** trees ek ke baad ek bante hain; naya tree pehle bane ensemble ki bachi hui error ko sudharne ki koshish karta hai.
- **Gradient:** error/loss ko kam karne ki direction batane wali mathematical information.

### Boosting ko example se samjho

Socho pehla tree historical patterns se forecast banata hai, par kuch dates par uski prediction galat hai. Agla tree un galtiyon ko dekhkar correction seekhta hai. Phir agla tree bachi hui error ko correct karta hai. Sab trees ke contributions judkar final output banta hai. Isliye **ek bada tree nahi**, balki corrections karte chhote trees ka group hota hai.

Training ka loss yahan return prediction ki error ko kam karne ki koshish karta hai. Final quality phir bhi rupee-price MAPE/MAE aur baseline ke against evaluate hoti hai.

### LightGBM is project ke liye practical kyun hai?

- Data ek **tabular/numeric table** hai: har row ek date aur columns engineered features hain.
- Trees nonlinear threshold pakad sakte hain; price behavior high-volatility aur low-volatility period mein alag ho sakta hai.
- Features ko linear model ki tarah same scale par lana aam taur par zaroori nahi.
- Is size ke engineered dataset par train karna tez aur prediction API mein serve karna aasaan hai.
- Regularization/tree limits se complexity ko control kiya ja sakta hai.

Yeh reasons LightGBM ko **reasonable candidate** banate hain; iska matlab yeh nahi ki yeh har dataset par best hai. Project ke actual evidence ke liye holdout comparison dekhna chahiye.

### `train.py` ke LightGBM parameters ka matlab

Current implementation mein parameters:

| Parameter | Current value | Simple meaning / kyun |
|---|---:|---|
| `n_estimators` | 400 | Maximum boosting trees/rounds. Early stopping ho to actual count kam ho sakta hai. |
| `learning_rate` | 0.03 | Har naye tree ka correction kitna dheere add ho. Chhota rate stable learning de sakta hai, par zyada rounds chahiye. |
| `num_leaves` | 15 | Har tree mein maximum leaf/end-nodes. Kam leaves = simpler tree, overfit ka risk kam. |
| `max_depth` | 4 | Tree kitni levels deep ho sakti hai. Depth limit complexity ko rokta hai. |
| `min_child_samples` | 10 | Leaf banane ke liye minimum training examples. Chhote random groups par rules banne se rokta hai. |
| `subsample` | 0.85 | Har boosting round mein rows ka subset use karne ka setting. Isse variability/overfit kam ho sakta hai. |
| `colsample_bytree` | 0.85 | Har tree mein features ka subset. Har tree ko bilkul same clues par dependent hone se rokta hai. |
| `reg_alpha` | 0.15 | L1 regularization penalty; unnecessarily complex/large effects ko discourage karta hai. |
| `reg_lambda` | 0.15 | L2 regularization penalty; weights/effects ko shrink karke overfitting ko control karne mein madad. |
| `random_state` | 42 | Randomized steps ko repeatable banane ke liye fixed seed. |

**Early stopping:** Training validation tail par check karta hai. Agar 35 rounds tak improvement nahi aaye, LightGBM aur trees add karna rok sakta hai. Isse validation par bekaar training jaari rakhne se bachne ki koshish hoti hai.

**Regularization/overfitting simple meaning:** Overfitting tab hota hai jab model training examples ko bahut yaad kar leta hai, par unseen future par achha nahi chalta. Depth/leaves/minimum-samples limits, regularization, validation aur baseline test is risk ko control/check karne ke tareeqe hain—khatam karne ki guarantee nahi.

### LightGBM ke fayde aur limits

**Fayde:** tabular data par strong candidate, nonlinear patterns, feature scaling ki kam zaroorat, fast training/inference, tree size control.

**Limits:** chhote/noisy dataset par overfit ho sakta hai; naya weather, arrivals ya policy shock apne aap nahi jaan sakta; extrapolation yani dekhe hue price range se bahut bahar prediction tree models ka weak point ho sakta hai; feature importance causation prove nahi karti. LightGBM ka naam bhar accuracy ka proof nahi—holdout aur baseline evidence chahiye.

### Fallback model

Agar LightGBM install nahi ho, script scikit-learn `HistGradientBoostingRegressor` use karti hai. Yeh bhi histogram-based gradient-boosted trees hai aur similar limits lagata hai. Environment mein kaunsa actual model chala, training log/installed package se confirm karna chahiye; code mein dono paths maujood hain.

## 6. LightGBM ko doosron se kyun choose kiya—aur kya claim nahi karna

| Model | Is project mein default kyun nahi | Kab test karna useful |
|---|---|---|
| Linear Regression | Simple/interpretable, par nonlinear thresholds/interactions miss kar sakti hai. | Transparent benchmark ke roop mein zaroor. |
| Random Forest | Nonlinear tree model hai, par current feature table ke liye boosting practical candidate tha. Humne ise is report mein benchmark nahi kiya. | Same time split aur tuning ke saath compare karein. |
| ARIMA/SARIMA | Trend/seasonality time-series baseline ke liye useful; har crop/horizon ka setup alag aur careful chahiye. | Statistical time-series comparator ke roop mein. |
| LSTM/GRU/Transformer | Sequence neural networks; zyada data, tuning aur compute lag sakta hai. Sparse daily mandi records par overfit risk. Is project ne head-to-head test nahi kiya. | Jab zyada years/markets/external data ho, same chronological test par. |
| Persistence/Moving Average | Bahut simple, par stable prices mein surprisingly strong. | Hamesha benchmark rakhein; current system mein yeh production fallback bhi hai. |

Teacher ko yeh accurate statement dein: **“LightGBM ko structured data aur nonlinear patterns ke liye practical candidate choose kiya. Humne ise simple baseline se time-based test par compare kiya. Humne abhi har algorithm se comparison karke LightGBM ko universal winner prove nahi kiya.”**

## 7. Evaluation metrics: score ka matlab

- **MAE (Mean Absolute Error):** actual aur predicted price ke absolute difference ka average; unit yahan ₹/quintal. Chhota MAE better.
- **MAPE (Mean Absolute Percentage Error):** `average(abs(actual − predicted) / actual) × 100`. Misaal: actual ₹1,000 aur forecast ₹1,050 ho to us row ka error 5%. Chhota MAPE better. Actual zero/near-zero ho to MAPE problematic hota hai; pipeline positive prices hi rakhti hai.
- **Direction accuracy:** kitni baar model ne sahi kaha ki price current se upar, neeche ya same hoga. Direction sahi ho par rupee amount bahut galat ho sakta hai; isliye akela metric nahi.
- **Walk-forward CV:** teen rounds mein pehle past block par train aur uske baad ke block par test; har round mein train history aage expand hoti hai. Horizon gap rehta hai. Isse alag historical periods par stability dekhte hain.
- **Residual quantile range:** holdout errors ke 10th aur 90th percentile se low/high estimate banta hai. Yeh historical errors se nikla approximate band hai, guarantee ya har condition mein formally calibrated confidence interval nahi.

## 8. Current saved report humein kya batati hai?

Project ke current `accuracy_report.csv` mein 15 crop/horizon rows hain. Inka average model MAPE lagbhag **8.71%**, selected best-baseline MAPE ka average **8.11%** hai. ML model **4/15** configurations mein best baseline se behtar hai; API **11/15** mein baseline use karne ke liye marked hai.

Iska seedha matlab: har crop/horizon par complex ML zaroori nahi ki simple forecast se achha ho. Agar prediction current price ke barabar dikhe, ho sakta hai API ne persistence baseline chuna ho. Crop aur horizon ke hisaab se alag metrics batana average number se zyada informative hai. Tomato jaise crops mein error zyada ho sakta hai.

**Dates ka nuance:** current model metadata mein learned weights ka training-through date late 2025 hai. `model_history.csv` mein refreshed quotes Green Chilli, Potato, Rice, Tomato ke liye 2026-10-21 tak aur Wheat ke liye 2026-05-04 tak hain. Nayi history file model weights ko automatically retrain nahi karti. Recent retraining candidate ka average MAPE ~9.53% aaya, jo purane saved set se kharab tha; isliye purane weights rakhe gaye. Live current quote, API history aur model ke last training date alag cheezein hain.

## 9. Limitations aur agle sudhaar

1. **Forecast guarantee nahi:** mausam, supply/arrivals, transport, demand, MSP/policy jaise external changes input mein nahi, isliye model unhe predict nahi kar sakta.
2. **Market definition:** current prepared data sabhi district mandiyon ka confirmed combined average nahi. Desired target—ek mandi ya district-wide aggregate—pehle define karein.
3. **Cleaning mein future information ka risk:** current outlier filter centered 15-day median use karta hai; interpolation later endpoint se earlier gap fill kar sakti hai. Stronger research evaluation ke liye trailing-only outlier rule aur har forecast date par sirf us waqt available data use karke rerun karna chahiye.
4. **Model/baseline selection:** current holdout par model aur baseline compare hote hain aur isi result se winner choose hota hai. Final research score ke liye earlier validation par winner choose karein aur alag untouched final test period bachayein.
5. **Market selection bias:** kai markets compare karke same validation period ka best market choose karne se score optimistic ho sakta hai. Naye later period par confirm karein.
6. **External features nahi:** future version mein arrivals/volume, weather, holidays, fuel/transport, official policy data test ho sakta hai—par har input us date par available hona chahiye, warna leakage hoga.

Limitations batana achhi research practice hai. Iska matlab project bekaar hai nahi; iska matlab result ko uski evidence ke hisaab se samajhna hai.

## 10. Teacher ke likely questions (viva answers)

**Q: Modal price hi kyun?**  
A: Yeh source ka representative traded price hai aur app/API isi price ko show karna chahta hai. Isliye target Modal_Price hai; minimum aur maximum alag range values hain.

**Q: Har crop ka alag model kyun?**  
A: Crops ki seasonality, perishability, storage aur price behavior alag hai. Ek hi learned pattern sab crops par force nahi karte.

**Q: 1, 7, 14 din ke alag models kyun?**  
A: Forecast horizon ke saath target aur uncertainty badalte hain. Har horizon apne future return se train hota hai.

**Q: Log return kyun?**  
A: Yeh rupee amount ke bajay proportional change sikhata hai, different price levels ko compare karne mein madad karta hai, aur exponential conversion positive forecast deta hai.

**Q: LightGBM kya hai ek line mein?**  
A: Gradient-boosted decision trees ka fast framework—chhote trees ek ke baad ek pichhli errors correct karte hain, aur sabka output milkar prediction deta hai.

**Q: Boosting aur Random Forest mein basic fark?**  
A: Boosting mein trees sequentially pichhli errors sudharte hain. Random Forest mein aam taur par kai trees alag-alag train hote hain aur unki predictions average/vote hoti hain. Yeh general distinction hai; dono ko hamare data par compare kiye bina winner claim nahi karenge.

**Q: `learning_rate` kam kyun?**  
A: Har naye tree ka correction chhota rakhta hai; learning gradual hoti hai. Iske liye zyada boosting rounds ki zaroorat ho sakti hai.

**Q: `num_leaves`, `max_depth`, `min_child_samples` kya karte hain?**  
A: Tree ki complexity aur leaf banane ke liye required examples ko limit karte hain. Isse training rows ko memorize karne ka risk kam ho sakta hai.

**Q: Overfitting kya hai?**  
A: Training examples par bahut achha aur unseen future data par kharab chalna. Time-based holdout, walk-forward, early stopping, regularization aur baselines is risk ko check/control karte hain.

**Q: Recency weighting kya hai?**  
A: Recent records ko zyada influence aur purane ko kam influence dena; yahan weight teen saal ke decay scale par kam hota hai.

**Q: Random split kyun nahi?**  
A: Forecasting past se future hoti hai. Random split future date ka pattern train mein la sakta hai aur unrealistic score de sakta hai.

**Q: Embargo kya hai?**  
A: Train/test boundary par horizon jitne din ka gap, taaki training target ka future test window mein na ghuse.

**Q: MAPE aur MAE mein fark?**  
A: MAE rupees mein average absolute error batata hai; MAPE percentage mein. Dono error size dekhte hain, alag units mein.

**Q: Direction accuracy kaafi kyun nahi?**  
A: Yeh sirf up/down/same direction check karta hai, movement kitne rupees ka tha nahi. Isliye MAE/MAPE bhi chahiye.

**Q: Prediction current price ke equal kyun?**  
A: Is crop/horizon ke holdout mein “price same rahega” baseline ML se kam MAPE de sakta hai. Metadata ke basis par API persistence baseline chunti hai.

**Q: LightGBM ko best prove kiya?**  
A: Nahi. Current code mein LightGBM candidate ka simple baselines se comparison hai; har possible model se head-to-head test nahi. Isliye ise practical choice kehte hain, universal best nahi.

**Q: LightGBM ki weakness?**  
A: Noisy/chhote data par overfit ho sakta hai; unseen external shock nahi jaanta; bahut out-of-range price extrapolate karna iska strong point nahi.

**Q: Kya confidence band guaranteed hai?**  
A: Nahi. Yeh past holdout residual percentiles se approximate band hai; future market conditions alag ho sakti hain.

**Q: Sabse important limitation?**  
A: Model mainly past modal prices aur calendar features dekhta hai. Market scope/cleaning bhi exact forecast target se match honi chahiye; current version mein causal cleaning aur untouched final test behtar karne ke areas hain.

**Q: Agla improvement kya hoga?**  
A: Causal cleaning, exact mandi/district aggregation, validation se baseline choice aur alag final test, phir weather/arrivals jaise time-aligned external features aur alternative models ko same split par compare karna.

## 11. Presentation mein bolne ke liye conclusion

> Humne LightGBM ko isliye practical candidate chuna kyunki hamara data structured numerical features wala hai aur trees nonlinear patterns seekh sakte hain. Har crop aur forecast horizon ke liye alag target banaya. Future dates ko test mein rakha, horizon embargo lagaya, aur persistence/moving-average baselines se compare kiya. Current report mein LightGBM 15 mein se 4 configurations mein baseline se achha hai, isliye API baaki cases mein simple baseline chunti hai. Agla kaam cleaning ko fully past-only banana, exact market aggregation confirm karna aur untouched final test par validate karna hai.

## 12. Training dobara chalane ka command

```powershell
python train.py --input ghaziabad_best_mandi_2018_2026.csv --district Ghaziabad --market Ghaziabad --start-date 2018-01-01 --end-date 2026-10-25
```

Naye artifacts deploy karne se pehle `accuracy_report.csv` dekhein. Current deployed files ko tab tak overwrite na karein jab tak candidate ka evaluation review aur promotion decision na ho. Files promote karne ke baad backend restart karein, taaki API unhe reload kare.
