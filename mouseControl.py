import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

#Timers
click_timer_started = False
click_start_time = 0

back_timer_started = False
back_start_time = 0

prev_x, prev_y = 0, 0
smooth_factor = 0.1  # quanto menor, mais suave (0.1 ~ 0.3 são bons valores)


def finger_up(lm, tip_id, pip_id):
    return lm[tip_id][1] < lm[pip_id][1]

def get_hand_landmarks(hand_landmark, img_w, img_h):
    return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in hand_landmark.landmark]

def is_closed_fist(landmarks):
    return all(landmarks[i][1] > landmarks[i - 2][1] for i in [8, 12, 16, 20])

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img_height, img_width, _ = img.shape

    right_hand = None
    left_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_handedness in enumerate(results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            hand_landmarks = results.multi_hand_landmarks[i]
            landmarks = get_hand_landmarks(hand_landmarks, img_width, img_height)

            if hand_label == "Right":
                right_hand = landmarks
            else:
                left_hand = landmarks

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # === GESTOS ===

    # MOVIMENTO: dedo indicador da mão direita adaptado para 21:9 compacto
    if right_hand and finger_up(right_hand, 8, 6):
        x, y = right_hand[8]
    
        # Proporção 16:9 e escala da janela de controle
        aspect_ratio = 16 / 9
        scale = 0.4  #40% da área total
    
        # Calcula a largura e altura da área de controle compacta
        area_width = int(img_width * scale)
        area_height = int(area_width / aspect_ratio)
    
        # Centraliza a área na imagem
        x1 = (img_width - area_width) // 2
        x2 = x1 + area_width
        y1 = (img_height - area_height) // 2
        y2 = y1 + area_height
    
        # Só move o mouse se o dedo estiver dentro da área útil
        if x1 <= x <= x2 and y1 <= y <= y2:
            screen_x = np.interp(x, [x1, x2], [0, screen_width])
            screen_y = np.interp(y, [y1, y2], [0, screen_height])
            # Suavização do movimento
            curr_x = screen_x * (1 - smooth_factor) + prev_x * smooth_factor
            curr_y = screen_y * (1 - smooth_factor) + prev_y * smooth_factor

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            cv2.putText(img, 'MOVER (compactado)', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
        # (Visualização) Desenha a área de controle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 2)




    # CLIQUE: todos os dedos esticados por 3 segundos
    if right_hand:
        all_fingers_up = all(finger_up(right_hand, tip, tip - 2) for tip in [8, 12, 16, 20])
        thumb_up = right_hand[4][0] < right_hand[3][0]  # Polegar aberto (mais à esquerda que o ponto da articulação)

        if all_fingers_up and thumb_up:
            if not click_timer_started:
                click_timer_started = True
                click_start_time = time.time()
            elif time.time() - click_start_time > 1:
                pyautogui.click()
                cv2.putText(img, 'CLIQUE!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                click_timer_started = False
        else:
            click_timer_started = False

        if click_timer_started:
            elapsed = time.time() - click_start_time
            cv2.putText(img, f'CLIQUE EM: {1 - int(elapsed)}s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 255, 150), 2)



  # SCROLL UP
    if right_hand:
        pinky_up = finger_up(right_hand, 20, 18)
        others_down = not any(finger_up(right_hand, i, i-2) for i in [8, 12, 16])
        if pinky_up and others_down:
            pyautogui.scroll(80)  # valor maior = scroll mais rápido
            cv2.putText(img, 'SCROLL UP', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # SCROLL DOWN
    if right_hand:
        thumb_up = finger_up(right_hand, 4, 3)
        others_down = not any(finger_up(right_hand, i, i-2) for i in [8, 12, 16, 20])
        if thumb_up and others_down:
            pyautogui.scroll(-80)
            cv2.putText(img, 'SCROLL DOWN', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)



    # VOLTAR PÁGINA: dedo médio levantado por 3 segundos
    if right_hand:
        middle_up = finger_up(right_hand, 12, 10)
        other_fingers_down = not any(finger_up(right_hand, i, i-2) for i in [8, 16, 20])

        if middle_up and other_fingers_down:
            if not back_timer_started:
                back_timer_started = True
                back_start_time = time.time()
            elif time.time() - back_start_time > 2:
                pyautogui.hotkey('alt', 'left')
                cv2.putText(img, 'VOLTAR PÁGINA', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
                back_timer_started = False
        else:
            back_timer_started = False

        if back_timer_started:
            elapsed = time.time() - back_start_time
            cv2.putText(img, f'VOLTAR EM: {2 - int(elapsed)}s', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 200, 255), 2)

    # Mostrar a imagem
    cv2.imshow("Controle com Gestos", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC pra sair
        break

cap.release()
cv2.destroyAllWindows()
