import json
from flask import Flask, request, jsonify
from chatarena.arena import Arena

app = Flask(__name__)
app.arena = None

def filter_observations(observations, player_name):
    filtered_observations = []
    for observation in observations:
        #print(observation)
        visible_to = observation.visible_to
        if visible_to == "all" or player_name in visible_to:
            filtered_observations.append(observation)
            visible_to = "all"
    return filtered_observations

@app.route('/', methods=['GET'])
def health():
    return jsonify({'msg': "ok"})

@app.route('/chatarena/setup', methods=['POST'])
def setup_game():
    # 创建Arena对象
    i_env_conf = "whoisthespy"
    data = request.get_json()
    if 'env_conf' in data:
        i_env_conf = data.get('env_conf', "whoisthespy")
    
    # env_conf = 'whoisthespy'
    app.arena = Arena.from_config(f'examples/{i_env_conf}.json')
    return jsonify({'msg': f"arena {i_env_conf} is loaded!"})

@app.route('/chatarena/reset', methods=['POST'])
def reset_game():
    if app.arena is None:
        return jsonify({'error': "arena is undefined!"}), 400
    app.arena.reset()
    app.arena = None
    return jsonify({'msg': "arena is unloaded!"})

@app.route('/chatarena/p_step', methods=['POST'])
def p_step_game():
    try:
        data = request.get_json()
        i_player_name = ""
        i_player_action = ""
        if 'player_name' in data:
            i_player_name = data.get('player_name', "")
        if 'player_action' in data:
            i_player_action = data.get('player_action', "")
        
        timestep = app.arena.step(i_player_name, i_player_action)
        observation = timestep.observation
        terminal = timestep.terminal
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400  

    # 返回结果
    response = {
        'observation': observation,
        'terminal': terminal
    }
    
    new_response = dict(response)

    new_response["observation"] = filter_observations(new_response["observation"], i_player_name)
#     output["observation"] = data
    return jsonify(new_response)

#curl -X POST http://localhost:8080/chatarena/step
@app.route('/chatarena/step', methods=['POST'])
def step_game():
    if app.arena is None:
        return jsonify({'error': "arena is undefined!"}), 400

    try:
        data = request.get_json()
        i_player_name = data.get('player_name', "")
        i_player_action = data.get('player_action', "")
        
        timestep = app.arena.step(i_player_name, i_player_action)
        observation = timestep.observation
        terminal = timestep.terminal
        # if timestep.terminal:
        #     raise Exception("arena is end!")
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # 返回结果
    response = {
        'observation': observation,
        'terminal': terminal
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
