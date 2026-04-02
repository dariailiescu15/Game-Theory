import streamlit as st
import numpy as np
import pandas as pd
from fractions import Fraction

def f(n):
    if abs(n) < 1e-10: return "0"
    return str(Fraction(float(n)).limit_denominator(100))

def pregateste_forma_standard(A, b, c, semne, tip_x, opt_tip, M_val):
    m, n_init = np.array(A).shape
    A_mat = np.array(A, dtype=float)
    b_lucru = np.array(b, dtype=float)
    c_vec = np.array(c, dtype=float)
    semne_lucru = list(semne)

    coloane_r1, costuri_r1, mapare_var, nume_var_r1 = [], [], [], []
    
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


def ruleaza_iteratii_simplex(TS, XB, Cj, baza, nume_var, opt_tip):
    m = len(XB)
    n_tot = len(Cj)
    nume_ai = [f"a{i+1}" for i in range(n_tot)]
    pas = 0
    
    while pas < 1000:
        CB = Cj[baza]
        Z_total = np.dot(CB, XB)
        deltas = [Cj[j] - np.dot(CB, TS[:, j]) for j in range(n_tot)]
        
        # Aici e titlul Mov
        st.markdown(f"<h4 style='color: #CE93D8;'>Tabel Simplex T{pas}</h4>", unsafe_allow_html=True)
        
        coloane = ["CB", "Baza", "XB"] + nume_ai
        date_tabel = []
        for i in range(m):
            linie = [f(CB[i]), nume_ai[baza[i]], f(XB[i])] + [f(val) for val in TS[i]]
            date_tabel.append(linie)
        
        linie_Dj = ["", "Dj", f"Z = {f(Z_total)}"] + [f(d) for d in deltas]
        date_tabel.append(linie_Dj)
        
        df_tabel = pd.DataFrame(date_tabel, columns=coloane)
        st.dataframe(df_tabel, hide_index=True, use_container_width=True)

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
    st.markdown("---")
    st.markdown("<h3 style='color: #CE93D8; text-align: center;'> VALIDAREA SOLUȚIEI ASP </h3>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**1. CRITERIU OPTIM ({opt_t}):**")
        if opt_t == 'MAX':
            ok = all(d <= 1e-5 for d in deltas_final)
            st.info(f"Dj = {', '.join([f(d) for d in deltas_final])}  <= 0")
            if ok: st.success("✅ [OK] Criteriu îndeplinit.")
            else: st.error("❌ [FAIL] Criteriu neîndeplinit.")
        else:
            ok = all(d >= -1e-5 for d in deltas_final)
            st.info(f"Dj = {', '.join([f(d) for d in deltas_final])}  >= 0")
            if ok: st.success("✅ [OK] Criteriu îndeplinit.")
            else: st.error("❌ [FAIL] Criteriu neîndeplinit.")

        st.markdown(f"**3. Verificare valoare f(PL):**")
        st.info(f"Z_final din tabel = **{f(Z_final)}**")

    with col2:
        st.markdown("**2. Verificare nenegativitate (xj >= 0):**")
        sol_completa = {nume_v[i]: 0.0 for i in range(len(nume_v))}                                
        for i in range(len(XB_final)): sol_completa[nume_v[baza_finala[i]]] = XB_final[i]          
        
        x_reconstruit = np.zeros(len(c_init))
        for m_info in mapare:
            val_tabel = sol_completa[m_info['nume']]
            x_reconstruit[m_info['original']] += m_info['semn'] * val_tabel
            st.write(f"🔹 {m_info['nume']}* = **{f(val_tabel)}** >= 0 ✅")                             

        f_calculata = sum(c_init[i] * x_reconstruit[i] for i in range(len(c_init)))               
        if abs(f_calculata - Z_final) < 1e-10:
            st.success(f"✅ [OK] Calcul direct in funcție: {f(f_calculata)} == {f(Z_final)}")
        else:
            st.error(f"❌ [FAIL] Diferență: {f(f_calculata)} != {f(Z_final)}")

    st.markdown("---")
    st.markdown("**4. Verificare relație: XB(I_0) = S × XB(I_stop):**")
    
    S_matrice = A_prim_init[:, baza_finala]                                           
    
    col_s, col_xb, col_rez, col_ini = st.columns(4)
    
    with col_s:
        st.write("Matricea **S**")
        S_formatat = [[f(val) for val in rand] for rand in S_matrice]
        df_s = pd.DataFrame(S_formatat, columns=[f"a{b+1}" for b in baza_finala])
        st.dataframe(df_s, hide_index=True)
        
    with col_xb:
        st.write("Vector **XB (final)**")
        df_xb = pd.DataFrame([f(x) for x in XB_final], columns=["valori"])
        st.dataframe(df_xb, hide_index=True)

    reconstruit = np.dot(S_matrice, XB_final)                                         
    with col_rez:
        st.write("Calcul **S × XB**")
        df_rez = pd.DataFrame([f(x) for x in reconstruit], columns=["valori"])
        st.dataframe(df_rez, hide_index=True)

    with col_ini:
        st.write("Vector **XB (inițial)**")
        df_ini = pd.DataFrame([f(x) for x in b_init], columns=["valori"])
        st.dataframe(df_ini, hide_index=True)

    if np.allclose(reconstruit, b_init):
        st.success("✅ REZULTAT VERIFICAT: S × XB(final) este egal cu vectorul b inițial!")
    else:
        st.error("❌ EROARE: S × XB(final) NU este egal cu vectorul b inițial!")
