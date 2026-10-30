from predict import predict

print("=== Testing Tomato 7d with different input prices ===")
for p in [500, 1000, 2000, 4000]:
    r = predict("Tomato", 7, as_of_date="today", current_price=p)
    pp = r["predicted_price"]
    cp = r["change_percent"]
    adv = r["advice"]
    print(f"  Input: Rs.{p:5d}  -> Forecast: Rs.{pp:8.2f}  ({cp:+.2f}%)  | Advice: {adv}")

print()
print("=== Testing Green Chilli 14d with different input prices ===")
for p in [2000, 4000, 6000, 8000]:
    r = predict("Green Chilli", 14, as_of_date="today", current_price=p)
    pp = r["predicted_price"]
    cp = r["change_percent"]
    adv = r["advice"]
    print(f"  Input: Rs.{p:5d}  -> Forecast: Rs.{pp:8.2f}  ({cp:+.2f}%)  | Advice: {adv}")

print()
print("=== Testing Wheat 7d with different input prices ===")
for p in [2000, 2500, 3000, 3500]:
    r = predict("Wheat", 7, as_of_date="today", current_price=p)
    pp = r["predicted_price"]
    cp = r["change_percent"]
    adv = r["advice"]
    print(f"  Input: Rs.{p:5d}  -> Forecast: Rs.{pp:8.2f}  ({cp:+.2f}%)  | Advice: {adv}")
