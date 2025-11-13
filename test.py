import numpy as np

class AdaptiveKalmanFilterAltitude:
    """
    Filtre de Kalman Adaptatif pour l'estimation d'altitude (GPS seul) 
    avec ajustement de Q basé sur l'activité (pour le Trail).
    """

    def __init__(self, dt, initial_altitude):
        # --- 1. État initial [altitude, vitesse] ---
        self.state = np.array([[initial_altitude], [0.0]])
        
        # --- 2. Matrice de transition d'état (A) ---
        self.A = np.array([[1.0, dt], 
                           [0.0, 1.0]])
        
        # --- 3. Matrice de mesure (H) ---
        self.H = np.array([[1.0, 0.0]]) 
        
        # --- 4. Covariance de l'erreur d'état (P) ---
        self.P = np.array([[100.0, 0.0], 
                           [0.0, 10.0]])
                           
        # --- 5. Covariance du bruit de mesure (R) FIXE (GPS BRUITÉ) ---
        # On suppose que l'incertitude du GPS est constante et élevée (entre 10 et 30 m^2).
        self.R = np.array([[20.0]]) 
        
        # --- 6. Base de Covariance du bruit de processus (Q) ---
        # Petites valeurs de base pour une faible activité (Q sera ajusté dynamiquement)
        self.Q_base = np.array([[0.001, 0.0], 
                                [0.0, 0.01]])
        
        # --- 7. Paramètres d'adaptation ---
        self.ADAPT_FACTOR = 0.5  # Facteur de sensibilité (ajuster entre 0.1 et 1.0)
        self.MAX_Q_VEL = 0.5     # Vitesse max permise pour Q (pour éviter la divergence)


    def adjust_Q_for_trail(self):
        """ Ajuste la matrice Q en fonction de la vitesse verticale estimée. """
        
        # Récupère la vitesse verticale estimée (deuxième élément du vecteur d'état)
        estimated_vertical_velocity = self.state[1, 0]
        
        # Utilise la valeur absolue de la vitesse verticale
        abs_velocity = np.abs(estimated_vertical_velocity)
        
        # Le facteur d'adaptation augmente si la vitesse est grande
        # Ce facteur détermine dans quelle mesure nous augmentons la confiance du modèle pour changer
        adaptation_scale = self.ADAPT_FACTOR * abs_velocity
        
        # Assurer que l'échelle ne dépasse pas une limite raisonnable
        adaptation_scale = np.clip(adaptation_scale, 0.01, self.MAX_Q_VEL)

        # Applique l'adaptation principalement à l'incertitude de la vitesse (Q[1, 1])
        Q_adapted = np.copy(self.Q_base)
        Q_adapted[1, 1] += adaptation_scale * adaptation_scale # Au carré pour un effet non-linéaire
        
        return Q_adapted

    def predict(self):
        """ Étape de prédiction avec Q adapté """
        
        # Obtenir la matrice Q adaptée
        Q_k = self.adjust_Q_for_trail()

        # Prédiction de l'état: x_k|k-1 = A * x_k-1|k-1
        self.state = self.A @ self.state 
        
        # Prédiction de la covariance: P_k|k-1 = A * P_k-1|k-1 * A.T + Q_k (adapté)
        self.P = self.A @ self.P @ self.A.T + Q_k
        
        return self.state[0, 0]

    def update(self, z):
        """ Étape de mise à jour (Correction avec la mesure z) """
        
        # Innovation (Erreur de prédiction de mesure): y = z - H * x_k|k-1
        y = z - self.H @ self.state
        
        # Covariance d'innovation: S = H * P_k|k-1 * H.T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Gain de Kalman: K = P_k|k-1 * H.T * S.inv
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Correction de l'état: x_k|k = x_k|k-1 + K * y
        self.state = self.state + K @ y
        
        # Correction de la covariance: P_k|k = (I - K * H) * P_k|k-1
        self.P = (np.eye(self.A.shape[0]) - K @ self.H) @ self.P
        
        return self.state[0, 0]

# --- Exemple d'Utilisation ---
# Simule une course de trail : montée, descente rapide, puis plat avec bruit GPS
dt = 1.0 
montagne_descente = [
    50, 51, 53, 56, 60, 65, 68, 70, 71, 72, 
    70, 65, 58, 52, 50, 49, 48, 48, 48, 48
]
np.random.seed(42)
mesures_altitude_gps = [m + np.random.normal(0, 3.5) for m in montagne_descente] # Bruit GPS plus élevé

# Initialisation du filtre adaptatif
initial_altitude = mesures_altitude_gps[0]
kf_adapt = AdaptiveKalmanFilterAltitude(dt=dt, initial_altitude=initial_altitude)

altitudes_filtrees = []

# Boucle de filtrage
for mesure in mesures_altitude_gps:
    kf_adapt.predict()
    altitude_filtree = kf_adapt.update(mesure)
    altitudes_filtrees.append(altitude_filtree)
    
print("--- Résultats du Filtrage Adaptatif ---")
print(f"Altitudes filtrées (début) : {np.round(altitudes_filtrees[:10], 2)}")
print(f"Altitudes filtrées (fin) : {np.round(altitudes_filtrees[10:], 2)}")

# --- Les lignes suivantes montrent les résultats ---
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(mesures_altitude_gps, 'r.', alpha=0.5, label='Altitude GPS Brute (Mesure)')
plt.plot(altitudes_filtrees, 'b-', linewidth=2, label='Altitude Filtrée (Kalman Adaptatif)')
plt.plot(montagne_descente, 'g--', label='Altitude "Réelle" (Simulée)')
plt.title('Filtre de Kalman Adaptatif pour l\'Altitude en Trail')
plt.xlabel('Étape de temps')
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)
plt.show()