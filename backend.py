from fastapi import FastAPI
from pydantic import BaseModel
from ortools.sat.python import cp_model

app = FastAPI()

class ScheduleRequest(BaseModel):
    workers: list[str]
    shifts: list[str]
    H: dict[str, int]  # hours required per shift
    M: dict[str, int]  # max hours per worker
    avail: dict[str, int] 
    forbidden_pairs: list[list[str]] = []  # JSON usa liste, non tuple

@app.post("/schedule")
def schedule(data: ScheduleRequest):
    workers, shifts, H, M = data.workers, data.shifts, data.H, data.M

    # converti "Alice,MonLunch" -> ("Alice","MonLunch")
    avail = {}
    for key, v in data.avail.items():
        w, s = key.split(",")
        avail[(w.strip(), s.strip())] = v

    forbidden_pairs = [tuple(fp) for fp in data.forbidden_pairs]

    # ---- OR-Tools model ----
    model = cp_model.CpModel()
    x = {}

    for w in workers:
        for s in shifts:
            if avail.get((w, s), 0) == 1:
                x[(w, s)] = model.NewBoolVar(f"x_{w}_{s}")
            else:
                x[(w, s)] = model.NewConstant(0)

    for w in workers:
        for (s1, s2) in forbidden_pairs:
            model.Add(x[(w, s1)] + x[(w, s2)] <= 1)

    for s in shifts:
        model.Add(sum(x[(w, s)] for w in workers) == 1)

    for w in workers:
        model.Add(sum(x[(w, s)] * H[s] for s in shifts) <= M[w])

    model.Maximize(sum(x[(w, s)] * H[s] for w in workers for s in shifts))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    res = solver.Solve(model)

    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = {"total_hours": solver.ObjectiveValue(), "assignments": {}}
        for w in workers:
            assigned = []
            hours = 0
            for s in shifts:
                if solver.Value(x[(w, s)]) == 1:
                    assigned.append({"shift": s, "hours": H[s]})
                    hours += H[s]
            result["assignments"][w] = {"tasks": assigned, "hours": hours}
        return result
    else:
        return {"error": "No feasible assignment found"}
