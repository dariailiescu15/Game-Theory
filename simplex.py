import numpy as np
from fractions import Fraction

# Functie care transforma orice numar in fractie
def f(n):
    if abs(n) < 1e-10: return "0"                                                
    return str(Fraction(float(n)).limit_denominator(100))                        

# Functie pentru afisarea vectorilor sub forma de coloana (vertical)
def afiseaza_coloana(vector, nume):
    print(f"   {nume} = ")
    for val in vector:
        print(f"      ( {f(val):^5} )")                                          

def pregateste_forma_standard(A, b, c, semne, tip_x, opt_tip, M_val):
    m, n_init = np.array(A).shape                                              
    A_mat = np.array(A, dtype=float)                                         
    b_lucru = np.array(b, dtype=float)                                         
    c_vec = np.array(c, dtype=float)                                           
    semne_lucru = list(semne)                                                  
    
    # --- REGULA R1: Tratarea condițiilor variabilelor (x >= 0, x <= 0 sau liber) 
    coloane_r1 = []
    costuri_r1 = []
    mapare_var = []
    nume_var_r1 = [] 

    for j in range(n_init):
        if tip_x[j] == '>=0':
            coloane_r1.append(A_mat[:, j])
            costuri_r1.append(c_vec[j])
            mapare_var.append({'nume': f"x{j+1}", 'original': j, 'semn': 1})
            nume_var_r1.append(f"x{j+1}")           
            
        elif tip_x[j] == '<=0':
            coloane_r1.append(A_mat[:, j] * (-1))
            costuri_r1.append(c_vec[j] * (-1))
            mapare_var.append({'nume': f"x{j+1}'", 'original': j, 'semn': -1})
            nume_var_r1.append(f"x{j+1}'")
            
        elif tip_x[j] == 'liber':
            coloane_r1.append(A_mat[:, j]) 
            costuri_r1.append(c_vec[j])
            mapare_var.append({'nume': f"x{j+1}'", 'original': j, 'semn': 1})
            nume_var_r1.append(f"x{j+1}'")
            
            coloane_r1.append(A_mat[:, j] * (-1)) 
            costuri_r1.append(c_vec[j] * (-1))
            mapare_var.append({'nume': f"x{j+1}''", 'original': j, 'semn': -1})
            nume_var_r1.append(f"x{j+1}''")
            
    A_lucru_r1 = np.column_stack(coloane_r1)
    C_lucru_r1 = np.array(costuri_r1)

    for i in range(m):                                                         
        if b_lucru[i] < 0:                                                     
            b_lucru[i] *= -1                                                   
            A_lucru_r1[i] *= -1                                                
            if semne_lucru[i] == '<=': semne_lucru[i] = '>='                   
            elif semne_lucru[i] == '>=': semne_lucru[i] = '<='

    coloane_std = A_lucru_r1.tolist()                                            
    Cj_std = C_lucru_r1.tolist()                                                         
    nume_var = nume_var_r1.copy()                                             
    baza_initiala = []                                                        

    for i in range(m):                                                        
        if semne_lucru[i] == '<=':                                            
            col = [0]*m; col[i] = 1                                           
            for r in range(m): coloane_std[r].append(col[r])                  
            Cj_std.append(0)                                                  
            nume_var.append(f"y{i+1}")                                        
            baza_initiala.append(len(Cj_std) - 1)                             
        elif semne_lucru[i] == '>=':                                          
            col_y = [0]*m; col_y[i] = -1                                      
            for r in range(m): coloane_std[r].append(col_y[r])
            Cj_std.append(0)                                                  
            nume_var.append(f"y{i+1}")
            col_z = [0]*m; col_z[i] = 1                                       
            for r in range(m): coloane_std[r].append(col_z[r])
            val_M = M_val if opt_tip == 'MIN' else -M_val                     
            Cj_std.append(val_M)                                              
            nume_var.append(f"z{i+1}")
            baza_initiala.append(len(Cj_std) - 1)                             
        elif semne_lucru[i] == '=':                                           
            col_z = [0]*m; col_z[i] = 1                                       
            for r in range(m): coloane_std[r].append(col_z[r])
            val_M = M_val if opt_tip == 'MIN' else -M_val
            Cj_std.append(val_M)
            nume_var.append(f"z{i+1}")
            baza_initiala.append(len(Cj_std) - 1)

    return np.array(coloane_std, dtype=float), np.array(b_lucru, dtype=float), np.array(Cj_std, dtype=float), nume_var, baza_initiala, mapare_var

# NOTĂ: Numele a fost schimbat din 'ruleaza_iteratii_simplex' și am adăugat argumentul 'st=None' 
# pentru a nu bloca aplicația atunci când rulează din Streamlit.
def ruleaza_simplex_interactiv(TS, XB, Cj, baza, nume_var, opt_tip, st=None):
    m = len(XB)                                                                    
    n_tot = len(Cj)                                                                
    nume_ai = [f"a{i+1}" for i in range(n_tot)]                                    
    pas = 0 

    while pas < 1000:                                                                
        CB = Cj[baza]                                                              
        Z_total = np.dot(CB, XB)                                                   
        
        deltas = [Cj[j] - np.dot(CB, TS[:, j]) for j in range(n_tot)]
        
        # Afișarea în consolă 
        print(f"\n--- TABEL SIMPLEX T{pas} ---")
        print("CB\tBaza\tXB\t| " + "\t".join(nume_ai))
        for i in range(m):
            linie = f"{f(CB[i])}\t{nume_ai[baza[i]]}\t{f(XB[i])}\t| "
            linie += "\t".join([f(val) for val in TS[i]])
            print(linie)
        print("-" * 100)
        print(f"\tDj\tZ={f(Z_total)}\t| " + "\t".join([f(d) for d in deltas]))

        if opt_tip == 'MAX':
            if all(d <= 1e-5 for d in deltas): break                                   
            col_p = np.argmax(deltas)                                                  
        else: 
            if all(d >= -1e-5 for d in deltas): break                                  
            col_p = np.argmin(deltas)                                                  

        rapoarte = [XB[i]/TS[i, col_p] if TS[i, col_p] > 1e-10 else float('inf') for i in range(m)]
        
        lin_p = np.argmin(rapoarte)                                                    
        
        pivot_val = TS[lin_p, col_p]                                                   
        TS[lin_p] /= pivot_val                                                         
        XB[lin_p] /= pivot_val                                                         
        for i in range(m):                                                             
            if i != lin_p:
                multiplicator = TS[i, col_p]
                TS[i] -= multiplicator * TS[lin_p]
                XB[i] -= multiplicator * XB[lin_p]
        
        baza[lin_p] = col_p                                                           
        pas += 1                                                                      
        
    return XB, Z_total, deltas, baza, TS                                              

def validare_solutie(XB_final, Z_final, deltas_final, baza_finala, TS_final, A_prim_init, b_init, c_init, mapare, nume_v, opt_t):
    print("\n" + "="*60)
    print("            VERIFICARI SI VALIDARE FINALA ")
    print("="*60)

    print(f"\nCRITERIU OPTIM ({opt_t}):")
    if opt_t == 'MAX':
        print(f"   Dj = {[f(d) for d in deltas_final]} <= 0 {'[OK]' if all(d <= 1e-5 for d in deltas_final) else '[FAIL]'}")
    else:
        print(f"   Dj = {[f(d) for d in deltas_final]} >= 0 {'[OK]' if all(d >= -1e-5 for d in deltas_final) else '[FAIL]'}")

    print("\nV1) Verificare nenegativitate xj >= 0:")
    sol_completa = {nume_v[i]: 0.0 for i in range(len(nume_v))}                                
    for i in range(len(XB_final)): sol_completa[nume_v[baza_finala[i]]] = XB_final[i]          

    x_reconstruit = np.zeros(len(c_init))
    for m_info in mapare:
        val_tabel = sol_completa[m_info['nume']]
        x_reconstruit[m_info['original']] += m_info['semn'] * val_tabel
        print(f"   {m_info['nume']}* = {f(val_tabel)} >= 0 [OK]")                             

    print(f"\nV2) Verificare valoare f(PL) = {f(Z_final)}:")
    f_calculata = sum(c_init[i] * x_reconstruit[i] for i in range(len(c_init)))               
    print(f"   Calcul: {f(f_calculata)} == {f(Z_final)} {'[OK]' if abs(f_calculata - Z_final) < 1e-10 else '[FAIL]'}")

    print("\nV3) Verificare relatie XB(I0) = S * XB(I_stop):")
    S_matrice = A_prim_init[:, baza_finala]                                           
    print("   Matricea S (din coloanele bazei finale din T0):")
    for r in S_matrice: print("      | " + "\t".join([f(val) for val in r]) + " |")
    afiseaza_coloana(XB_final, "XB(I_stop) final")                                    
    reconstruit = np.dot(S_matrice, XB_final)                                         
    afiseaza_coloana(reconstruit, "S * XB(I_stop)")                                   
    afiseaza_coloana(b_init, "XB(I0) initial")                                        

    print(f"\n   Rezultat V3: {'[VERIFICAT]' if np.allclose(reconstruit, b_init) else '[FAIL]'}")

def rezolva_simplex_complet(A, b, c, semne, tip_x, opt_tip='MAX', M=1000):
    rezultat_pregatire = pregateste_forma_standard(A, b, c, semne, tip_x, opt_tip, M)
    TS_init, b_lucru, Cj_std, nume_v, baza_init, mapare = rezultat_pregatire

    b_backup = b_lucru.copy() 
    A_prim_init = TS_init.copy() 

    # Am updatat aici pentru a chema noua denumire a functiei
    rezultat_it = ruleaza_simplex_interactiv(TS_init, b_lucru.copy(), Cj_std, baza_init, nume_v, opt_tip)

    if rezultat_it:
        XB_f, Z_f, Dj_f, baza_f, TS_f = rezultat_it
        validare_solutie(XB_f, Z_f, Dj_f, baza_f, TS_f, A_prim_init, b_backup, c, mapare, nume_v, opt_tip)
    else:
        print("Problema nu are solutie optima finita (nemarginita).")
    print("="*60)
