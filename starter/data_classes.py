
from dataclasses import dataclass, field
from enum import Enum
import json 
import numpy as np
from peft import PeftModel
import random
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List


def calculate_accuracy_metrics(results: List[Dict]):
    """
    Calculates accuracy metrics from evaluation results.
    
    Args:
        results: A list of dictionaries containing evaluation results
        
    Returns:
        Dictionary with accuracy percentage, total scenarios, and correct count
    """

    if not results:
        return {'accuracy': 0.0, 
                'total_scenarios': 0, 
                'correct_count': 0}
    correct_count = sum(r.get('is_correct', 0) for r in results)
    total_scenarios = len(results)
    accuracy = (correct_count / total_scenarios) * 100 if total_scenarios > 0 else 0
    return {'accuracy': accuracy, 
            'total_scenarios': total_scenarios, 
            'correct_count': correct_count}
def get_true_outcome(ground_truth_probability: float) -> str:
    """
    Gets the outcome from the probability.
    
    Args:
        ground_truth_probability: A float expressing the probability produced by the model
        
    Returns:
        "YES" if probability > 0.5, "NO" otherwise
    """


    return "YES" if ground_truth_probability > 0.5 else "NO"

class PersonType(Enum):
    JUNIOR_FOUNDER = "junior_founder"
    SENIOR_FOUNDER = "senior_founder"
    JUNIOR_ENGINEER = "junior_engineer"
    SENIOR_ENGINEER = "senior_engineer"
    VC_PARTNER = "vc_partner"
    VC_ANALYST = "vc_analyst"
    SALES_EXEC = "sales_exec"
    PRODUCT_MANAGER = "product_manager"
    RESEARCHER = "researcher"
    EXECUTIVE = "executive"

class TimeSlot(Enum):
    DAY1_MORNING = "day1_morning"
    DAY1_AFTERNOON = "day1_afternoon"
    DAY1_EVENING = "day1_evening"
    DAY2_MORNING = "day2_morning"
    DAY2_AFTERNOON = "day2_afternoon"
    DAY2_EVENING = "day2_evening"
    DAY3_MORNING = "day3_morning"
    DAY3_AFTERNOON = "day3_afternoon"
    DAY3_EVENING = "day3_evening"

@dataclass
class Person:
    id: int
    type: PersonType
    energy: float = 1.0
    introversion: float = 0.5
    seniority: float = 0.5
    meetings_had: int = 0
    successful_connections: int = 0
    meeting_history: List[int] = field(default_factory=list)
    mood: float = 0.5

@dataclass
class Meeting:
    person_a_id: int
    person_b_id: int
    time_slot: TimeSlot
    success: bool = False
    satisfaction: float = 0.0

class ConferenceSimulator:
    def __init__(self, num_attendees: int = 500, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.num_attendees = num_attendees
        self.attendees: Dict[int, Person] = self._generate_attendees()
        self.meetings_log: List[Meeting] = []
        self._hidden_coeffs = self._generate_hidden_coeffs(seed)

    def _generate_hidden_coeffs(self, seed: int) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        return {
            "energy_decay_base": rng.uniform(0.1, 0.2), "introvert_energy_multiplier": rng.uniform(1.5, 2.5),
            "extrovert_energy_gain_on_success": rng.uniform(0.05, 0.15), "mood_influence_strength": rng.uniform(0.1, 0.3),
            "seniority_nervousness_penalty": rng.uniform(0.2, 0.4), "repeat_meeting_value_decay": rng.uniform(0.6, 0.8),
            "network_boost_strength": rng.uniform(0.05, 0.15), "base_noise_std": rng.uniform(0.02, 0.08),
            "morning_person_bonus": rng.uniform(0.1, 0.2), "evening_introvert_penalty": rng.uniform(0.15, 0.3),
            "day_fatigue_penalty": rng.uniform(0.05, 0.1),
        }

    def _generate_attendees(self) -> Dict[int, Person]:
        attendees = {}
        type_distribution = {
            PersonType.JUNIOR_FOUNDER: 0.15, PersonType.SENIOR_FOUNDER: 0.05, PersonType.JUNIOR_ENGINEER: 0.20,
            PersonType.SENIOR_ENGINEER: 0.10, PersonType.VC_PARTNER: 0.05, PersonType.VC_ANALYST: 0.10,
            PersonType.SALES_EXEC: 0.10, PersonType.PRODUCT_MANAGER: 0.15, PersonType.RESEARCHER: 0.05, PersonType.EXECUTIVE: 0.05,
        }
        for i in range(self.num_attendees):
            person_type = np.random.choice(list(type_distribution.keys()), p=list(type_distribution.values()))
            if "junior" in person_type.value: seniority = np.random.uniform(0.1, 0.4)
            elif "senior" in person_type.value or "partner" in person_type.value or person_type == PersonType.EXECUTIVE: seniority = np.random.uniform(0.7, 1.0)
            else: seniority = np.random.uniform(0.4, 0.7)
            if person_type in [PersonType.JUNIOR_ENGINEER, PersonType.SENIOR_ENGINEER, PersonType.RESEARCHER]: introversion = np.random.uniform(0.6, 0.9)
            elif person_type in [PersonType.SALES_EXEC]: introversion = np.random.uniform(0.1, 0.4)
            else: introversion = np.random.uniform(0.3, 0.7)
            attendees[i] = Person(id=i, type=person_type, energy=1.0, introversion=introversion, seniority=seniority, mood=0.5)
        return attendees
    
    def _get_base_type_chemistry(self, type_a: PersonType, type_b: PersonType) -> float:
        chemistry_matrix = {
            (PersonType.JUNIOR_FOUNDER, PersonType.VC_PARTNER): 0.3, (PersonType.JUNIOR_FOUNDER, PersonType.VC_ANALYST): 0.6,
            (PersonType.SENIOR_FOUNDER, PersonType.VC_PARTNER): 0.8, (PersonType.SENIOR_FOUNDER, PersonType.VC_ANALYST): 0.5,
            (PersonType.JUNIOR_FOUNDER, PersonType.SENIOR_ENGINEER): 0.7, (PersonType.SENIOR_FOUNDER, PersonType.SENIOR_ENGINEER): 0.6,
            (PersonType.JUNIOR_ENGINEER, PersonType.SENIOR_ENGINEER): 0.8, (PersonType.JUNIOR_ENGINEER, PersonType.JUNIOR_ENGINEER): 0.7,
            (PersonType.SALES_EXEC, PersonType.EXECUTIVE): 0.8, (PersonType.SALES_EXEC, PersonType.PRODUCT_MANAGER): 0.7,
            (PersonType.RESEARCHER, PersonType.RESEARCHER): 0.9, (PersonType.RESEARCHER, PersonType.JUNIOR_ENGINEER): 0.4,
        }
        key = tuple(sorted((type_a.value, type_b.value)))
        mapped_key = next((k for k in chemistry_matrix if tuple(sorted((k[0].value, k[1].value))) == key), None)
        if mapped_key: return chemistry_matrix[mapped_key]
        if type_a == type_b: return 0.7
        return 0.5

    def _calculate_meeting_success(self, person_a: Person, person_b: Person, time_slot: TimeSlot) -> Tuple[bool, float]:
        day = int(time_slot.value.split('_')[0][-1])
        time_of_day = time_slot.value.split('_')[1]
        f_base_chem = self._get_base_type_chemistry(person_a.type, person_b.type)
        f_energy = 1 / (1 + np.exp(-10 * (person_a.energy * person_b.energy - 0.2)))
        f_mood = 1 / (1 + np.exp(-5 * ((person_a.mood + person_b.mood) / 2 - 0.5)))
        penalty_a = np.exp(-person_a.meetings_had * person_a.introversion * self._hidden_coeffs["introvert_energy_multiplier"] * (1.5 if time_of_day == 'evening' else 1.0))
        penalty_b = np.exp(-person_b.meetings_had * person_b.introversion * self._hidden_coeffs["introvert_energy_multiplier"] * (1.5 if time_of_day == 'evening' else 1.0))
        f_introvert_fatigue = (penalty_a + penalty_b) / 2
        f_time_of_day = 1.0 + (self._hidden_coeffs["morning_person_bonus"] * (1 - (person_a.introversion + person_b.introversion) / 2) if time_of_day == "morning" else -self._hidden_coeffs["evening_introvert_penalty"] * ((person_a.introversion + person_b.introversion) / 2) if time_of_day == "evening" else 0)
        f_day_fatigue = 1.0 - (day - 1) * self._hidden_coeffs["day_fatigue_penalty"]
        f_seniority_mismatch = 1.0 - (self._hidden_coeffs["seniority_nervousness_penalty"] if abs(person_a.seniority - person_b.seniority) > 0.6 and ("junior" in person_a.type.value or "junior" in person_b.type.value) and time_of_day == "morning" and day == 1 else 0)
        f_memory = self._hidden_coeffs["repeat_meeting_value_decay"] if person_b.id in person_a.meeting_history else 1.0
        f_network_effect = 1 + len(set(person_a.meeting_history) & set(person_b.meeting_history)) * self._hidden_coeffs["network_boost_strength"]
        success_prob = np.clip((f_base_chem * f_energy * f_mood * f_introvert_fatigue * f_time_of_day * f_day_fatigue * f_seniority_mismatch * f_memory * f_network_effect) + np.random.normal(0, self._hidden_coeffs["base_noise_std"]), 0, 1)
        success = np.random.random() < success_prob*2
        return success, 1-success_prob if success else success_prob
    
    def _update_person_state(self, person: Person, meeting_success: bool, satisfaction: float, time_slot: TimeSlot):
        energy_cost = self._hidden_coeffs["energy_decay_base"] * (1 + person.introversion * self._hidden_coeffs["introvert_energy_multiplier"])
        if meeting_success and person.introversion < 0.3: energy_cost -= self._hidden_coeffs["extrovert_energy_gain_on_success"]
        person.energy = max(0, person.energy - energy_cost)
        person.meetings_had += 1
        person.mood = np.clip(person.mood * 0.7 + satisfaction * 0.3, 0, 1)
        if meeting_success and person.id not in person.meeting_history: person.meeting_history.append(person.id)
        day_current = int(time_slot.value.split('_')[0][-1])
        last_day = int(getattr(person, 'last_meeting_time_slot', time_slot.value).split('_')[0][-1])
        if day_current > last_day:
            person.energy = min(1.0, person.energy + 0.3 + np.random.uniform(0, 0.2) * (1 - person.introversion))
            person.mood = min(1.0, person.mood + 0.1 + np.random.uniform(0, 0.1))
        person.last_meeting_time_slot = time_slot.value
    
    def _calculate_metrics(self) -> Dict:
        if not self.meetings_log: return {"composite_score": 0}
        active_people = [p for p in self.attendees.values() if p.meetings_had > 0]
        if not active_people: return {"composite_score": 0}
        avg_satisfaction = np.mean([m.satisfaction for m in self.meetings_log])
        success_rate = np.mean([m.success for m in self.meetings_log])
        burnout_rate = np.mean([1 for p in self.attendees.values() if p.energy < 0.3])
        coverage = len(active_people) / self.num_attendees
        return {"composite_score": float(np.clip(avg_satisfaction * 0.4 + success_rate * 0.3 + (1 - burnout_rate) * 0.2 + coverage * 0.1, 0, 1))}



class PersonDescriptor:
    def __init__(self, temperature: float = 0.8):
        self.temperature = temperature
    
    def generate_description(self, person: Person, time_slot: TimeSlot) -> str:
        """Convert person attributes to natural language description"""
        prompt = f"""Describe this conference attendee's current state in 1-2 sentences based on their situation:

Role: {person.type.value.replace('_', ' ').title()}
Current energy level: {'High' if person.energy > 0.7 else 'Medium' if person.energy > 0.4 else 'Low'}
Social preference: {'Introverted' if person.introversion > 0.6 else 'Extroverted' if person.introversion < 0.4 else 'Balanced'}
Experience level: {'Senior' if person.seniority > 0.7 else 'Junior' if person.seniority < 0.4 else 'Mid-level'}
Meetings today: {person.meetings_had}
Time: {time_slot.value.replace('_', ' ').title()}
Mood: {'Positive' if person.mood > 0.6 else 'Neutral' if person.mood > 0.4 else 'Struggling'}

Write a brief description of how this person appears and behaves right now."""
        
        return npcpy_get_llm_response(prompt, temperature=0.9).get('response')

# Add training metrics tracking
@dataclass 
class TrainingMetrics:
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    
    def __str__(self):
        return f"Epoch {self.epoch}, Step {self.step}: Loss={self.loss:.4f}, LR={self.learning_rate:.2e}"



class MeetingPredictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it', trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained('google/gemma-3-270m-it', trust_remote_code=True)
        self.model = PeftModel.from_pretrained(base_model, model_path)
    
    def predict(self, person_a_desc: str, person_b_desc: str, time_slot: str) -> dict:
        prompt = f"Person A: {person_a_desc}. Person B: {person_b_desc}. Time: {time_slot}. What is the likelihood of a successful meeting? Respond with JSON: {{\"probability\": 0.XX, \"reason\": \"word\"}}"
        test = f'<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n'
        
        inputs = self.tokenizer(test, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_part = result.split("model")[-1].strip()
        return json.loads(json_part)


def safe_extract_json(response_text: str) -> dict:
    """Extract and parse JSON with proper error handling"""
    try:
        # Remove extra periods at the end of descriptions
        response_text = re.sub(r'\.\s*\.+', '.', response_text)
        
        # Try multiple extraction patterns
        patterns = [
            r'<start_of_turn>model\n(.*?)(?:<end_of_turn>|$)',
            r'model\n(.*?)(?:<end_of_turn>|$)',
            r'(\{.*?\})',  # Any JSON-like structure
        ]
        
        json_str = None
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                break
        
        if not json_str:
            raise ValueError(f"No JSON found in response: {response_text}")
        
        # Clean up common issues
        json_str = json_str.replace('<end_of_turn>', '').strip()
        json_str = re.sub(r'\.\s*$', '', json_str)  # Remove trailing periods
        
        parsed = json.loads(json_str)
        
        # Validate required fields
        if 'probability' not in parsed:
            raise ValueError(f"Missing 'probability' field in JSON: {parsed}")
        
        prob = float(parsed['probability'])
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Probability {prob} not in valid range [0,1]")
        
        return parsed
        
    except Exception as e:
        print(f"Failed to parse JSON from response: {response_text}")
        raise ValueError(f"Failed to parse JSON from response '{response_text}': {str(e)}")
    

def run_local_agent_evaluation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_scenarios: List[Dict],
    model_type: str
) -> List[Dict]:
    """
    Evaluates the fine tuned agent's performance on test scenarios.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_scenarios: List of test scenario dictionaries
        model_type: String identifier for the model type being evaluated
        
    Returns:
        List of evaluation result dictionaries
    """

    print(f"\n--- Running Full Local Evaluation for {model_type.upper()} Model ---")
    results = []
    def custom_llm_response(prompt, 
                            model_name=None, 
                            provider=None, 
                            format=None, 
                            **kwargs):
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if format == 'json':
            try:
                return {"response": json.loads(response_text)}
            except:
                return {"response": response_text}
        return {"response": response_text.strip()}
    
    for persona_idx, persona in enumerate(system_prompt_configurations):
        for scenario in test_scenarios:
            scenario_id = f"{scenario['scenario_id']}_p{persona_idx}"
            print(f"--- Scenario {scenario_id} ({model_type}, {persona['name']}) ---")
            true_outcome = get_true_outcome(scenario['ground_truth'])            
            original_get_llm_response = None
            try:
                import npcpy.llm_funcs
                original_get_llm_response = npcpy.llm_funcs.get_llm_response
                npcpy.llm_funcs.get_llm_response = custom_llm_response
                current_agent = NPC(
                    name=persona["name"].lower(),
                    primary_directive=persona["primary_directive"],
                    tools=TOOLS,
                ) 
                tool_loop = AgentToolLoop(current_agent, max_iterations=8)
                
                initial_prompt = f"""Your task is to decide if two people should meet. Use the available tools to gather information step-by-step. 

Person A: {scenario['p1_desc']}
Person B: {scenario['p2_desc']}
Time Slot: {scenario['ts_str']}

Begin your analysis by calling a tool."""
                
                loop_result = tool_loop.run_tool_loop(initial_prompt)
                final_rec_data = loop_result['final_recommendation']
                
                agent_outcome = "FAIL"
                if final_rec_data and 'recommendation' in final_rec_data:
                    agent_outcome = final_rec_data['recommendation'].upper()
                
                is_correct = 1 if agent_outcome == true_outcome else 0
                results.append({
                    'scenario_id': scenario_id,
                    'persona': persona['name'],
                    'is_correct': is_correct,
                    'agent_outcome': agent_outcome,
                    'true_outcome': true_outcome,
                })
                print(f"Scenario {scenario_id} complete. GT={true_outcome}, Agent said={agent_outcome}, Correct={bool(is_correct)}")
                
            finally:
                if original_get_llm_response:
                    npcpy.llm_funcs.get_llm_response = original_get_llm_response
    
    return results