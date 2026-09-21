#!/usr/bin/env python3
"""
Synthetic insurance dataset generator for Laya via OpenRouter.

Pipeline per example:
  target scheduling -> canonical claim (deterministic world model) -> planner
  -> writer (sees canonical facts only, never labels) -> independent critic
  -> repair/retry -> global dedupe -> stratified splits -> report

The canonical claim is sampled in Python from the configured distributions and
logical constraints, so the ground truth is deterministic and defensible. The
planner only enriches it into a rich descriptive world; the writer never sees a
single classifier label. Laya itself produces per-option probabilities at
inference time: this dataset deliberately stores discrete/ordinal targets only.

Quick start:
  export OPENROUTER_API_KEY='sk-or-v1-...'
  python generate_laya_claims_openrouter.py --init-config insurance.json
  python generate_laya_claims_openrouter.py --config insurance.json \
      --output-dir data/claims --count 1000
  python generate_laya_claims_openrouter.py --config insurance.json --dry-run
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, copy, datetime as dt, hashlib, json
import math, os, random, re, sys, threading, time, unicodedata
import urllib.error, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VERSION = "2.0.0"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_CONFIG = json.loads(r"""
{
  "dataset": {
    "name": "laya-insurance-auto-it-synthetic-v2",
    "schema_id": "claims-auto-it-v2",
    "language": "it",
    "locale": "it-IT",
    "jurisdiction": "Italia",
    "product": "assicurazione auto RCA / garanzie accessorie",
    "seed": 42017,
    "count": 1000,
    "splits": {
      "train": 0.82,
      "validation": 0.13,
      "demo": 0.05
    },
    "stratify_on": "event_type",
    "include_questions_in_records": true,
    "keep_generation_facts": true,
    "label_notation": "true|false",
    "target_probabilities": false,
    "probability_note": "Il dataset contiene ground truth discreto/ordinale. Le probabilita' per opzione sono prodotte da Laya a inference time e non vanno confuse con i target sintetici."
  },
  "openrouter": {
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "timeout_seconds": 180,
    "max_http_retries": 6,
    "retry_base_seconds": 2.0,
    "http_referer": "",
    "x_title": "Laya Synthetic Claims Generator",
    "structured_output_fallback": true,
    "structured_output_first": true
  },
  "models": {
    "planner": {
      "model": "openai/gpt-5.4",
      "temperature": 0.85,
      "max_tokens": 4000,
      "extra": {}
    },
    "writer": {
      "model": "~anthropic/claude-sonnet-latest",
      "temperature": 1.0,
      "max_tokens": 3000,
      "extra": {}
    },
    "critic": {
      "model": "~google/gemini-pro-latest",
      "temperature": 0.15,
      "max_tokens": 3000,
      "extra": {
        "reasoning": {
          "effort": "low"
        }
      }
    }
  },
  "generation": {
    "workers": 3,
    "max_plan_attempts": 3,
    "max_writer_repairs": 2,
    "min_realism_score": 85,
    "min_label_alignment_score": 95,
    "min_information_fidelity_score": 95,
    "min_style_quality_score": 80,
    "reject_label_leakage": true,
    "near_duplicate_hamming_distance": 6,
    "replacement_rounds": 8,
    "request_jitter_seconds": [
      0.0,
      0.35
    ],
    "prefer_rich_multilabel": true,
    "rich_multilabel_probability": 0.65
  },
  "controls": {
    "event_type": {
      "collision": 0.24,
      "parking_damage": 0.13,
      "theft": 0.08,
      "attempted_theft": 0.06,
      "fire": 0.04,
      "glass": 0.09,
      "weather": 0.09,
      "vandalism": 0.07,
      "animal_collision": 0.05,
      "road_hazard": 0.07,
      "other": 0.08
    },
    "difficulty": {
      "easy": 0.18,
      "medium": 0.38,
      "hard": 0.33,
      "borderline": 0.11
    },
    "channel": {
      "web_form": 0.24,
      "email": 0.16,
      "call_center_transcript": 0.2,
      "mobile_app": 0.13,
      "agent_note": 0.12,
      "chat": 0.07,
      "broker_email": 0.08
    },
    "style": {
      "clear_formal": 0.13,
      "normal_customer": 0.24,
      "colloquial": 0.14,
      "telegraphic": 0.1,
      "noisy_typos": 0.1,
      "long_with_distractors": 0.1,
      "indirect_implicit": 0.11,
      "very_short": 0.08
    },
    "damage_severity": {
      "0": 0.04,
      "1": 0.3,
      "2": 0.34,
      "3": 0.22,
      "4": 0.1
    }
  },
  "event_subtypes": {
    "collision": {
      "rear_end": 0.28,
      "intersection_collision": 0.2,
      "side_impact": 0.18,
      "single_vehicle": 0.14,
      "parked_vehicle_hit": 0.2
    },
    "parking_damage": {
      "parked_vehicle_hit": 0.55,
      "single_vehicle": 0.2,
      "other": 0.25
    },
    "theft": {
      "vehicle_theft": 0.55,
      "component_theft": 0.45
    },
    "attempted_theft": {
      "attempted_vehicle_theft": 0.6,
      "component_theft": 0.4
    },
    "fire": {
      "other": 1.0
    },
    "glass": {
      "windshield_chip": 0.42,
      "windshield_crack": 0.4,
      "other": 0.18
    },
    "weather": {
      "hail": 0.45,
      "flood": 0.2,
      "wind": 0.25,
      "other": 0.1
    },
    "vandalism": {
      "keying": 0.45,
      "intentional_damage": 0.55
    },
    "animal_collision": {
      "animal_impact": 1.0
    },
    "road_hazard": {
      "pothole": 0.5,
      "road_debris": 0.5
    },
    "other": {
      "other": 1.0
    }
  },
  "event_profiles": {
    "__default__": {
      "vehicles_involved": {
        "one": 0.5,
        "two": 0.2,
        "three_or_more": 0.03,
        "unknown": 0.27
      },
      "liability_context": {
        "no_third_party": 0.3,
        "insured_at_fault": 0.15,
        "third_party_at_fault": 0.15,
        "shared_fault": 0.05,
        "unidentified_third_party": 0.1,
        "undetermined": 0.25
      }
    },
    "collision": {
      "vehicles_involved": {
        "one": 0.18,
        "two": 0.6,
        "three_or_more": 0.12,
        "unknown": 0.1
      },
      "liability_context": {
        "insured_at_fault": 0.34,
        "third_party_at_fault": 0.3,
        "shared_fault": 0.16,
        "unidentified_third_party": 0.08,
        "no_third_party": 0.09,
        "undetermined": 0.03
      }
    },
    "parking_damage": {
      "vehicles_involved": {
        "one": 0.45,
        "two": 0.33,
        "unknown": 0.22
      },
      "liability_context": {
        "no_third_party": 0.28,
        "unidentified_third_party": 0.34,
        "third_party_at_fault": 0.3,
        "shared_fault": 0.03,
        "undetermined": 0.05
      }
    },
    "theft": {
      "vehicles_involved": {
        "one": 0.97,
        "unknown": 0.03
      },
      "liability_context": {
        "no_third_party": 0.86,
        "unidentified_third_party": 0.11,
        "undetermined": 0.03
      }
    },
    "attempted_theft": {
      "vehicles_involved": {
        "one": 0.95,
        "unknown": 0.05
      },
      "liability_context": {
        "no_third_party": 0.8,
        "unidentified_third_party": 0.17,
        "undetermined": 0.03
      }
    },
    "fire": {
      "vehicles_involved": {
        "one": 1.0
      },
      "liability_context": {
        "no_third_party": 0.88,
        "undetermined": 0.12
      }
    },
    "glass": {
      "vehicles_involved": {
        "one": 0.72,
        "two": 0.22,
        "unknown": 0.06
      },
      "liability_context": {
        "no_third_party": 0.62,
        "third_party_at_fault": 0.16,
        "unidentified_third_party": 0.1,
        "undetermined": 0.12
      }
    },
    "weather": {
      "vehicles_involved": {
        "one": 1.0
      },
      "liability_context": {
        "no_third_party": 1.0
      }
    },
    "vandalism": {
      "vehicles_involved": {
        "one": 0.96,
        "unknown": 0.04
      },
      "liability_context": {
        "no_third_party": 0.62,
        "unidentified_third_party": 0.32,
        "undetermined": 0.06
      }
    },
    "animal_collision": {
      "vehicles_involved": {
        "one": 0.92,
        "two": 0.05,
        "unknown": 0.03
      },
      "liability_context": {
        "no_third_party": 0.8,
        "undetermined": 0.14,
        "unidentified_third_party": 0.06
      }
    },
    "road_hazard": {
      "vehicles_involved": {
        "one": 0.93,
        "two": 0.04,
        "unknown": 0.03
      },
      "liability_context": {
        "no_third_party": 0.7,
        "third_party_at_fault": 0.12,
        "undetermined": 0.18
      }
    }
  },
  "event_location_profiles": {
    "__default__": {
      "urban_road": 0.4,
      "extra_urban_road": 0.2,
      "motorway": 0.1,
      "parking_lot": 0.15,
      "private_area": 0.05,
      "unknown": 0.1
    },
    "collision": {
      "urban_road": 0.42,
      "extra_urban_road": 0.22,
      "motorway": 0.12,
      "parking_lot": 0.16,
      "private_area": 0.04,
      "unknown": 0.04
    },
    "parking_damage": {
      "parking_lot": 0.62,
      "urban_road": 0.26,
      "private_area": 0.06,
      "unknown": 0.06
    },
    "theft": {
      "urban_road": 0.5,
      "parking_lot": 0.3,
      "private_area": 0.12,
      "unknown": 0.08
    },
    "attempted_theft": {
      "urban_road": 0.52,
      "parking_lot": 0.28,
      "private_area": 0.12,
      "unknown": 0.08
    },
    "fire": {
      "urban_road": 0.45,
      "parking_lot": 0.3,
      "private_area": 0.15,
      "unknown": 0.1
    },
    "glass": {
      "urban_road": 0.48,
      "parking_lot": 0.2,
      "motorway": 0.17,
      "extra_urban_road": 0.1,
      "unknown": 0.05
    },
    "weather": {
      "urban_road": 0.4,
      "extra_urban_road": 0.25,
      "parking_lot": 0.2,
      "private_area": 0.1,
      "unknown": 0.05
    },
    "vandalism": {
      "urban_road": 0.55,
      "parking_lot": 0.3,
      "unknown": 0.15
    },
    "animal_collision": {
      "extra_urban_road": 0.55,
      "motorway": 0.15,
      "urban_road": 0.2,
      "unknown": 0.1
    },
    "road_hazard": {
      "urban_road": 0.5,
      "extra_urban_road": 0.3,
      "motorway": 0.1,
      "unknown": 0.1
    }
  },
  "conditional_facts": {
    "glass_damage": {
      "__default__": 0.03,
      "glass": 1.0,
      "collision": 0.24,
      "vandalism": 0.28,
      "weather": 0.16,
      "road_hazard": 0.1,
      "parking_damage": 0.05,
      "other": 0.06
    },
    "fire_damage": {
      "__default__": 0.0,
      "fire": 1.0,
      "collision": 0.03,
      "other": 0.02
    },
    "theft_or_attempt": {
      "__default__": 0.0,
      "theft": 1.0,
      "attempted_theft": 1.0,
      "vandalism": 0.05,
      "other": 0.03
    },
    "weather_damage": {
      "__default__": 0.0,
      "weather": 1.0,
      "road_hazard": 0.12,
      "parking_damage": 0.04,
      "other": 0.03
    },
    "vandalism_damage": {
      "__default__": 0.0,
      "vandalism": 1.0,
      "parking_damage": 0.06,
      "attempted_theft": 0.1,
      "other": 0.05
    },
    "other_property_damage": {
      "__default__": 0.18,
      "collision": 0.22,
      "fire": 0.35,
      "weather": 0.3,
      "road_hazard": 0.25,
      "parking_damage": 0.12
    }
  },
  "damage_severity_bounds": {
    "__default__": [
      0,
      4
    ],
    "collision": [
      1,
      4
    ],
    "parking_damage": [
      1,
      3
    ],
    "theft": [
      2,
      4
    ],
    "attempted_theft": [
      1,
      3
    ],
    "fire": [
      3,
      4
    ],
    "glass": [
      1,
      2
    ],
    "weather": [
      1,
      3
    ],
    "vandalism": [
      1,
      3
    ],
    "animal_collision": [
      1,
      4
    ],
    "road_hazard": [
      1,
      3
    ]
  },
  "severity_tables": {
    "towing_needed_probability": {
      "0": 0.0,
      "1": 0.03,
      "2": 0.15,
      "3": 0.6,
      "4": 0.95
    },
    "vehicle_drivable_probability": {
      "0": 1.0,
      "1": 0.98,
      "2": 0.85,
      "3": 0.45,
      "4": 0.0
    },
    "occupants_stranded_probability": {
      "stranded_vehicle": 0.62,
      "mobile": 0.04
    },
    "unsafe_vehicle_condition_probability": {
      "stranded_vehicle": 0.55,
      "mobile": 0.05
    }
  },
  "injury_roles": {
    "false": {
      "none": 1.0
    },
    "true": {
      "insured_driver": 0.3,
      "insured_passenger": 0.18,
      "third_party_driver": 0.2,
      "third_party_passenger": 0.11,
      "pedestrian_or_cyclist": 0.09,
      "multiple": 0.09,
      "unknown": 0.03
    }
  },
  "injury_severity_roles": {
    "false": {
      "none": 1.0
    },
    "true": {
      "minor": 0.58,
      "moderate": 0.28,
      "severe": 0.14
    }
  },
  "insured_driver_injured_probability": {
    "false": 0.0,
    "insured_driver": 1.0,
    "third_party_driver": 0.0,
    "third_party_passenger": 0.0,
    "pedestrian_or_cyclist": 0.0,
    "insured_passenger": 0.12,
    "multiple": 0.75,
    "unknown": 0.4
  },
  "legal_issue_profiles": {
    "clean": {
      "none": 0.94,
      "liability_dispute": 0.03,
      "third_party_dispute": 0.01,
      "recovery_dispute": 0.02
    },
    "disputed": {
      "none": 0.3,
      "liability_dispute": 0.3,
      "third_party_dispute": 0.18,
      "uninsured_or_unidentified_party": 0.12,
      "recovery_dispute": 0.06,
      "other": 0.04
    },
    "injury": {
      "none": 0.42,
      "injury_dispute": 0.28,
      "liability_dispute": 0.16,
      "third_party_dispute": 0.1,
      "other": 0.04
    },
    "authority": {
      "none": 0.48,
      "authority_or_proceeding": 0.32,
      "liability_dispute": 0.12,
      "injury_dispute": 0.05,
      "other": 0.03
    }
  },
  "direct_compensation": {
    "compatible_probability": 1.0,
    "required": {
      "third_party_involved": true,
      "counterparty_identified": true,
      "counterparty_insurance_known": true,
      "liability_context_in": [
        "third_party_at_fault",
        "shared_fault"
      ],
      "event_type_in": [
        "collision",
        "parking_damage"
      ],
      "vehicles_involved_in": [
        "two",
        "three_or_more"
      ]
    }
  },
  "fact_enums": {
    "legal_issue_type": [
      "none",
      "liability_dispute",
      "third_party_dispute",
      "injury_dispute",
      "uninsured_or_unidentified_party",
      "authority_or_proceeding",
      "recovery_dispute",
      "other"
    ],
    "event_location_type": [
      "urban_road",
      "extra_urban_road",
      "motorway",
      "parking_lot",
      "private_area",
      "unknown"
    ],
    "insured_driver_injured": [
      false,
      true
    ],
    "glass_damage": [
      false,
      true
    ],
    "fire_damage": [
      false,
      true
    ],
    "theft_or_attempt": [
      false,
      true
    ],
    "weather_damage": [
      false,
      true
    ],
    "vandalism_damage": [
      false,
      true
    ],
    "other_property_damage": [
      false,
      true
    ],
    "occupants_stranded": [
      false,
      true
    ],
    "unsafe_vehicle_condition": [
      false,
      true
    ]
  },
  "target_distributions": {
    "event_subtype": {
      "__conditional__": "event_subtypes"
    },
    "event_location_type": {
      "__conditional__": "event_location_profiles"
    },
    "liability_context": {
      "__conditional__": "event_profiles"
    },
    "vehicles_involved": {
      "__conditional__": "event_profiles"
    },
    "counterparty_identified": {
      "false": 0.3,
      "true": 0.7
    },
    "counterparty_insurance_known": {
      "false": 0.42,
      "true": 0.58
    },
    "authority_involved": {
      "false": 0.62,
      "true": 0.38
    },
    "event_date_known": {
      "false": 0.14,
      "true": 0.86
    },
    "injury_reported": {
      "false": 0.78,
      "true": 0.22
    },
    "injured_party_role": {
      "__conditional__": "injury_roles"
    },
    "injury_severity": {
      "__conditional__": "injury_severity_roles"
    },
    "damage_severity": {
      "__control__": "damage_severity"
    },
    "legal_issue_type": {
      "__conditional__": "legal_issue_profiles"
    },
    "documents_complete": {
      "0": 0.16,
      "1": 0.45,
      "2": 0.39
    },
    "urgency": {
      "0": 0.62,
      "1": 0.29,
      "2": 0.09
    },
    "ambiguity_level": {
      "0": 0.34,
      "1": 0.42,
      "2": 0.24
    }
  },
  "human_review_rules": {
    "force_true_if": {
      "ambiguity_level": [
        2
      ],
      "injury_severity": [
        "severe"
      ],
      "legal_issue_type": [
        "authority_or_proceeding",
        "injury_dispute"
      ]
    },
    "base_probability_when_clean": 0.06
  },
  "report": {
    "multi_label_combos": [
      [
        "rca_candidate",
        "kasko_candidate"
      ],
      [
        "rca_candidate",
        "driver_injury_candidate"
      ],
      [
        "kasko_candidate",
        "roadside_assistance_candidate"
      ],
      [
        "rca_candidate",
        "kasko_candidate",
        "driver_injury_candidate",
        "roadside_assistance_candidate"
      ],
      [
        "theft_candidate",
        "vandalism_candidate"
      ],
      [
        "glass_candidate",
        "kasko_candidate"
      ],
      [
        "natural_events_candidate",
        "kasko_candidate"
      ],
      [
        "legal_protection_candidate",
        "needs_human_review"
      ],
      [
        "direct_compensation_candidate",
        "rca_candidate"
      ]
    ]
  },
  "questions": {
    "event_type": {
      "type": "choice",
      "instructions": "Classifica la tipologia principale del sinistro descritto.",
      "criteria": {
        "collision": "collisione o urto mentre il veicolo era in movimento o contro un ostacolo",
        "parking_damage": "veicolo trovato danneggiato mentre era parcheggiato o in sosta, senza dinamica di marcia",
        "theft": "furto riuscito del veicolo o di componenti rilevanti",
        "attempted_theft": "tentativo di furto non riuscito con eventuali danni",
        "fire": "incendio o principio d'incendio come causa principale",
        "glass": "rottura o danneggiamento di parabrezza, lunotto o altri cristalli come evento principale",
        "weather": "danno causato principalmente da grandine, vento, allagamento o altro evento atmosferico",
        "vandalism": "danno intenzionale provocato da terzi senza finalita' di furto",
        "animal_collision": "urto con animale come causa principale",
        "road_hazard": "danno causato principalmente da buca, detrito, ostacolo o dissesto stradale",
        "other": "evento non riconducibile in modo affidabile alle categorie precedenti"
      }
    },
    "event_subtype": {
      "type": "choice",
      "instructions": "Individua il sottotipo piu' specifico compatibile con la tipologia di evento.",
      "criteria": {
        "rear_end": "tamponamento o urto posteriore",
        "intersection_collision": "collisione in intersezione o mancata precedenza",
        "side_impact": "urto laterale o fiancata",
        "single_vehicle": "uscita autonoma di strada o urto contro ostacolo senza altri veicoli",
        "parked_vehicle_hit": "veicolo urtato mentre era in sosta",
        "hail": "danni da grandine",
        "flood": "danni da allagamento o acqua",
        "wind": "danni da vento forte o caduta di rami",
        "windshield_chip": "scheggiatura del parabrezza",
        "windshield_crack": "crepa o rottura del parabrezza o di altro cristallo",
        "vehicle_theft": "furto dell'intero veicolo",
        "component_theft": "furto di componenti, parti o accessori",
        "attempted_vehicle_theft": "tentativo di furto del veicolo non riuscito",
        "keying": "danneggiamento volontario con graffi o strisciate",
        "intentional_damage": "danneggiamento volontario di altro tipo",
        "animal_impact": "urto con animale",
        "pothole": "danno da buca o dissesto stradale",
        "road_debris": "danno da detrito od ostacolo sulla carreggiata",
        "other": "sottotipo non altrimenti classificabile"
      }
    },
    "liability_context": {
      "type": "choice",
      "instructions": "Qual e' il contesto di responsabilita' desumibile dalla dinamica descritta?",
      "criteria": {
        "insured_at_fault": "dinamica che indica responsabilita' prevalente dell'assicurato",
        "third_party_at_fault": "dinamica che indica responsabilita' prevalente di un terzo",
        "shared_fault": "responsabilita' concorsuale o ripartita",
        "no_third_party": "nessun terzo coinvolto nella dinamica",
        "unidentified_third_party": "terzo coinvolto ma non identificato",
        "undetermined": "responsabilita' non determinabile dalle informazioni disponibili"
      }
    },
    "vehicles_involved": {
      "type": "choice",
      "instructions": "Quanti veicoli risultano coinvolti nella dinamica?",
      "criteria": {
        "one": "un solo veicolo",
        "two": "due veicoli",
        "three_or_more": "tre o piu' veicoli",
        "unknown": "numero di veicoli non desumibile"
      }
    },
    "third_party_involved": {
      "type": "noul",
      "instructions": "E' coinvolto almeno un terzo, identificato o chiaramente presente nella dinamica?"
    },
    "counterparty_identified": {
      "type": "noul",
      "instructions": "La controparte e' identificata con dati sufficienti (nome o targa)?"
    },
    "counterparty_insurance_known": {
      "type": "noul",
      "instructions": "Sono disponibili i dati assicurativi della controparte?"
    },
    "authority_involved": {
      "type": "noul",
      "instructions": "Sono intervenute forze dell'ordine o autorita' sul sinistro?"
    },
    "event_date_known": {
      "type": "noul",
      "instructions": "La data o la finestra temporale dell'evento e' nota con ragionevole certezza?"
    },
    "injury_reported": {
      "type": "noul",
      "instructions": "Nel sinistro vengono riportate lesioni o possibili lesioni a persone?"
    },
    "injured_party_role": {
      "type": "choice",
      "instructions": "Chi risulta ferito o potenzialmente ferito? Usa multiple quando sono ferite piu' persone con ruoli diversi.",
      "criteria": {
        "none": "nessun ferito",
        "insured_driver": "conducente del veicolo assicurato",
        "insured_passenger": "passeggero del veicolo assicurato",
        "third_party_driver": "conducente del veicolo terzo",
        "third_party_passenger": "passeggero del veicolo terzo",
        "pedestrian_or_cyclist": "pedone o ciclista",
        "multiple": "piu' persone ferite con ruoli diversi",
        "unknown": "lesioni riportate ma ruolo non chiaro"
      }
    },
    "injury_severity": {
      "type": "choice",
      "instructions": "Qual e' la gravita' delle lesioni riportate?",
      "criteria": {
        "none": "nessuna lesione",
        "minor": "lesioni lievi o referto di pronto soccorso senza ricovero",
        "moderate": "lesioni con prognosi significativa",
        "severe": "lesioni gravi o ricovero prolungato"
      }
    },
    "vehicle_drivable": {
      "type": "noul",
      "instructions": "Il veicolo assicurato risulta utilizzabile e in grado di marciare?"
    },
    "towing_needed": {
      "type": "noul",
      "instructions": "Il veicolo richiede soccorso stradale, traino o rimozione?"
    },
    "damage_severity": {
      "type": "score",
      "instructions": "Quanto e' grave il danno al veicolo assicurato?",
      "criteria": [
        "nessun danno rilevante",
        "danno lieve",
        "danno moderato",
        "danno grave",
        "perdita totale"
      ]
    },
    "rca_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia RCA (danno a terzi o responsabilita' verso terzi)?"
    },
    "kasko_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia Kasko (danni al veicolo assicurato)?"
    },
    "driver_injury_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la copertura infortuni del conducente?"
    },
    "legal_protection_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la tutela legale?"
    },
    "roadside_assistance_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante l'assistenza stradale?"
    },
    "glass_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia cristalli?"
    },
    "theft_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia furto?"
    },
    "fire_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia incendio?"
    },
    "natural_events_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia eventi naturali o atmosferici?"
    },
    "vandalism_candidate": {
      "type": "noul",
      "instructions": "La dinamica rende plausibilmente rilevante la garanzia atti vandalici?"
    },
    "direct_compensation_candidate": {
      "type": "noul",
      "instructions": "Il caso presenta i prerequisiti descrittivi tipici per una gestione a risarcimento diretto?"
    },
    "documents_complete": {
      "type": "score",
      "instructions": "Quanto e' completa la documentazione per avviare la gestione ordinaria?",
      "criteria": [
        "insufficiente",
        "parziale",
        "sufficiente"
      ]
    },
    "urgency": {
      "type": "score",
      "instructions": "Valuta la priorita' operativa del caso.",
      "criteria": [
        "gestione ordinaria",
        "gestione prioritaria a breve",
        "attenzione umana immediata"
      ]
    },
    "needs_human_review": {
      "type": "noul",
      "instructions": "Il caso richiede revisione umana per ambiguita', contraddizioni, severita' o eccezioni?"
    },
    "ambiguity_level": {
      "type": "score",
      "instructions": "Quanto e' ambiguo o contraddittorio il quadro informativo?",
      "criteria": [
        "chiaro",
        "alcune incertezze",
        "fortemente ambiguo"
      ]
    }
  }
}
""")


# --------------------------------------------------------------------------
# Canonical claim world model
# --------------------------------------------------------------------------

CANONICAL_FIELD_ORDER: Tuple[str, ...] = (
    "event_type", "event_subtype", "event_location_type",
    "liability_context", "vehicles_involved",
    "third_party_involved", "counterparty_identified",
    "counterparty_insurance_known", "authority_involved", "event_date_known",
    "injury_reported", "injured_party_role", "injury_severity",
    "insured_driver_injured",
    "vehicle_drivable", "towing_needed", "damage_severity",
    "glass_damage", "fire_damage", "theft_or_attempt", "weather_damage",
    "vandalism_damage", "other_property_damage",
    "occupants_stranded", "unsafe_vehicle_condition",
    "legal_issue_type",
    "documents_complete", "urgency", "ambiguity_level",
)

# Coverage / workflow decisions derived from the canonical claim. These are the
# only keys the writer must never observe.
LABEL_IDS: Tuple[str, ...] = (
    "rca_candidate", "kasko_candidate", "driver_injury_candidate",
    "legal_protection_candidate", "roadside_assistance_candidate",
    "glass_candidate", "theft_candidate", "fire_candidate",
    "natural_events_candidate", "vandalism_candidate",
    "direct_compensation_candidate", "needs_human_review",
)

KASKO_EVENTS = frozenset({
    "collision", "parking_damage", "glass", "weather", "vandalism",
    "fire", "animal_collision", "road_hazard", "other",
})
RCA_EVENTS = frozenset({"collision", "parking_damage"})
THEFT_EVENTS = frozenset({"theft", "attempted_theft"})
DRIVER_ROLES = frozenset({"insured_driver", "multiple"})
NO_DRIVER_ROLES = frozenset({"none", "insured_passenger", "third_party_driver",
                             "third_party_passenger", "pedestrian_or_cyclist"})

LEAK_TOKENS = frozenset(
    set(LABEL_IDS)
    | {"candidate", "label", "labels", "ground_truth", "targets", "question",
       "questions", "dataset", "synthetic", "prompt", "coverage_candidate"}
)

PLANNER_SYSTEM = """You design FICTIONAL insurance training cases. You receive a FIXED canonical claim (the authoritative world model of one claim) and you must expand it into rich, realistic, self-consistent descriptive material for the given jurisdiction/product.

Hard rules:
- Reproduce every canonical claim value EXACTLY. Never add, drop or contradict a canonical value.
- The descriptive facts must make every canonical claim value inferable by a careful human reader, without naming any internal taxonomy, any coverage/garanzia name, or any decision identifier.
- Never write the words candidate, RCA, kasko, tutela legale, risarcimento diretto, garanzia, polizza coperta, label or target.
- Use no real people, real plates, real claim IDs, or proprietary insurer forms.
- Difficulty may add distractors, indirect evidence, omissions and controlled contradictions, but the canonical claim must stay defensible.
- Output only structured JSON."""

WRITER_SYSTEM = """You write realistic FICTIONAL insurance claim records from canonical facts. You never receive classifier labels and you must never guess or name any coverage, guarantee or decision.

Preserve every material fact and every declared missing information. Invent no material fact. Never mention labels, decision IDs, targets, prompts, datasets, synthetic generation, coverage names or guarantees. Match the requested channel and style. Hard cases may be indirect, messy, colloquial, typo-ridden or full of distractors but must stay defensible by a careful human reader. Output only structured JSON."""

CRITIC_SYSTEM = """You are an independent strict auditor of a synthetic insurance dataset. Verify that the GENERATED STATE ITSELF supports every intended decision, not merely the hidden canonical claim. You must check explicitly and one by one:
- every decision is supported by the narrative text alone;
- no coverage candidate is positive without compatible facts in the text;
- no coverage candidate is negative while the narrative contains strong contrary evidence;
- injured party role is coherent with the text;
- liability context is coherent with the described dynamics;
- vehicle drivability and towing need are coherent with the described damage;
- legal protection is coherent with the presence or absence of a real dispute;
- direct compensation prerequisites are coherent with the described parties;
- there is no label leakage: no coverage, guarantee, decision, candidate or label wording;
- the writer invented no material fact;
- the writer dropped no material fact.
Ambiguity may make a case difficult but never unknowable. Reject anything mechanically templated. Return concise repair instructions. Output only structured JSON."""


def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def pretty(obj): return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
def now(): return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def stable_seed(seed, *parts):
    h = hashlib.sha256("|".join(map(str, (seed,)+parts)).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7fffffff

def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else copy.deepcopy(v)
    return out

def load_config(path):
    if not path: return copy.deepcopy(DEFAULT_CONFIG)
    text = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        try: import yaml
        except ImportError: raise SystemExit("YAML requires PyYAML; use JSON or pip install pyyaml")
        user = yaml.safe_load(text) or {}
    else: user = json.loads(text)
    return deep_merge(DEFAULT_CONFIG, user)

def dump_json(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":"))+"\n")

def append_jsonl(path, row, lock=None):
    line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n"
    if lock:
        with lock:
            with Path(path).open("a", encoding="utf-8") as f: f.write(line)
    else:
        with Path(path).open("a", encoding="utf-8") as f: f.write(line)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

def validate_config(cfg) -> List[str]:
    """Structural validation of the dataset configuration."""
    issues: List[str] = []
    qs = cfg.get("questions", {})
    if not qs: issues.append("questions is empty")
    for qid, q in qs.items():
        if q.get("type") not in ("choice", "score", "noul"):
            issues.append(f"{qid}: unknown type {q.get('type')!r}")
        if q["type"] == "choice" and not isinstance(q.get("criteria"), Mapping):
            issues.append(f"{qid}: choice requires a criteria map")
        if q["type"] == "score" and not isinstance(q.get("criteria"), Sequence):
            issues.append(f"{qid}: score requires a criteria list")

    for field in CANONICAL_FIELD_ORDER:
        if field not in qs and field not in cfg.get("fact_enums", {}):
            issues.append(f"canonical field {field} missing from questions and fact_enums")
    for qid in LABEL_IDS:
        if qid not in qs: issues.append(f"label question {qid} missing")
        elif qs[qid]["type"] != "noul": issues.append(f"label question {qid} must be noul")

    for key in ("difficulty", "channel", "style", "event_type", "damage_severity"):
        if key not in cfg.get("controls", {}): issues.append(f"controls.{key} missing")
    for ctl, table in (("event_type", cfg.get("event_subtypes", {})),
                       ("event_type", cfg.get("event_location_profiles", {})),
                       ("event_type", cfg.get("event_profiles", {})),
                       ("event_type", cfg.get("damage_severity_bounds", {})),
                       ("event_type", cfg.get("conditional_facts", {}) or {})):
        pass
    for et in cfg.get("controls", {}).get("event_type", {}):
        if et not in cfg.get("event_subtypes", {}): issues.append(f"event_subtypes[{et}] missing")
        if not profile_for(cfg.get("event_profiles", {}), et): issues.append(f"event_profiles[{et}] missing")
        if not profile_for(cfg.get("event_location_profiles", {}), et):
            issues.append(f"event_location_profiles[{et}] missing")
        if not profile_for(cfg.get("damage_severity_bounds", {}), et):
            issues.append(f"damage_severity_bounds[{et}] missing")
        for flag, table in cfg.get("conditional_facts", {}).items():
            if et not in table and "__default__" not in table:
                issues.append(f"conditional_facts[{flag}][{et}] missing")
        for sub in cfg.get("event_subtypes", {}).get(et, {}):
            if qs.get("event_subtype") and sub not in qs["event_subtype"]["criteria"]:
                issues.append(f"event_subtypes[{et}][{sub}] not in event_subtype criteria")

    dist = cfg.get("target_distributions", {})
    for qid, spec in dist.items():
        if isinstance(spec, Mapping) and ("__conditional__" in spec or "__control__" in spec): continue
        q = qs.get(qid)
        if q is None: issues.append(f"target_distributions[{qid}] has no question")
        elif q["type"] == "choice":
            for v in spec:
                if v not in q["criteria"]: issues.append(f"target_distributions[{qid}][{v}] not a legal choice")
        elif q["type"] == "score":
            for v in spec:
                if not (0 <= int(v) < len(q["criteria"])): issues.append(f"target_distributions[{qid}][{v}] out of range")

    for role in cfg.get("injury_roles", {}).get("true", {}):
        if role not in qs.get("injured_party_role", {}).get("criteria", {}):
            issues.append(f"injury_roles.true[{role}] not a legal injured_party_role")
    for sev in cfg.get("injury_severity_roles", {}).get("true", {}):
        if sev not in qs.get("injury_severity", {}).get("criteria", {}):
            issues.append(f"injury_severity_roles.true[{sev}] not a legal injury_severity")
    for lit in cfg.get("legal_issue_profiles", {}).get("clean", {}):
        if lit not in cfg.get("fact_enums", {}).get("legal_issue_type", []):
            issues.append(f"legal_issue_profiles.clean[{lit}] not a legal legal_issue_type")
    req = cfg.get("direct_compensation", {}).get("required", {})
    for key in ("event_type_in", "liability_context_in", "vehicles_involved_in"):
        for v in req.get(key, []):
            qid = {"event_type_in": "event_type", "liability_context_in": "liability_context",
                   "vehicles_involved_in": "vehicles_involved"}[key]
            if v not in qs[qid]["criteria"]: issues.append(f"direct_compensation.required.{key}: illegal {v}")
    return issues


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------

def pick_weighted(weights: Mapping[str, Any], rng: random.Random) -> str:
    items = [(str(k), float(v)) for k, v in weights.items() if float(v) > 0]
    if not items: raise ValueError("empty distribution")
    total = sum(v for _, v in items)
    x = rng.random() * total
    acc = 0.0
    for k, v in items:
        acc += v
        if x <= acc: return k
    return items[-1][0]

def pick_bool(dist: Mapping[str, Any], rng: random.Random) -> bool:
    return pick_weighted(dist, rng).lower() == "true"

def profile_for(table: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key in table: return table[key]
    return table.get("__default__", {})

def weighted_schedule(weights, n, rng):
    vals = [(str(k), max(0., float(v))) for k, v in weights.items() if float(v) > 0]
    total = sum(v for _, v in vals); raw = [(k, n*v/total) for k, v in vals]
    counts = {k: int(math.floor(x)) for k, x in raw}
    rem = n - sum(counts.values())
    frac = sorted([(x-math.floor(x), rng.random(), k) for k, x in raw], reverse=True)
    for _, _, k in frac[:rem]: counts[k] += 1
    out = []
    for k, _ in vals: out += [k]*counts[k]
    rng.shuffle(out); return out


# --------------------------------------------------------------------------
# Deterministic canonical claim sampling + logical constraints
# --------------------------------------------------------------------------

def sample_legal_issue(cfg, rng, f) -> str:
    if not f["third_party_involved"]:
        return "none"
    dispute = (f["liability_context"] in ("shared_fault", "undetermined")
               or not f["counterparty_identified"])
    if f["injury_reported"]:
        prof = "injury"
    elif dispute and f["authority_involved"]:
        prof = "authority"
    elif dispute:
        prof = "disputed"
    elif f["authority_involved"]:
        prof = "authority"
    else:
        prof = "clean"
    lit = pick_weighted(cfg["legal_issue_profiles"][prof], rng)
    if lit == "injury_dispute" and not f["injury_reported"]:
        lit = "liability_dispute"
    if lit == "authority_or_proceeding" and not f["authority_involved"]:
        lit = "liability_dispute"
    if lit == "uninsured_or_unidentified_party" and f["counterparty_identified"]:
        lit = "liability_dispute"
    return lit


def sample_world(cfg, rng, controls) -> Dict[str, Any]:
    qs = cfg["questions"]
    dist = cfg["target_distributions"]
    tab = cfg["severity_tables"]
    f: Dict[str, Any] = {}

    ev = controls["event_type"]
    f["event_type"] = ev

    bounds = cfg["damage_severity_bounds"]
    lo, hi = [int(x) for x in bounds.get(ev, bounds["__default__"])]
    f["damage_severity"] = max(lo, min(hi, int(controls["damage_severity"])))

    f["event_subtype"] = pick_weighted(cfg["event_subtypes"][ev], rng)
    f["event_location_type"] = pick_weighted(profile_for(cfg["event_location_profiles"], ev), rng)

    prof = profile_for(cfg["event_profiles"], ev)
    f["liability_context"] = pick_weighted(prof["liability_context"], rng)
    f["vehicles_involved"] = pick_weighted(prof["vehicles_involved"], rng)

    f["injury_reported"] = pick_bool(dist["injury_reported"], rng)
    key = "true" if f["injury_reported"] else "false"
    f["injured_party_role"] = pick_weighted(cfg["injury_roles"][key], rng)
    f["injury_severity"] = pick_weighted(cfg["injury_severity_roles"][key], rng)
    if not f["injury_reported"]:
        f["injured_party_role"] = "none"
        f["injury_severity"] = "none"
    f["insured_driver_injured"] = bool(
        f["injury_reported"] and f["injured_party_role"] in DRIVER_ROLES)

    f["third_party_involved"] = bool(
        f["liability_context"] != "no_third_party"
        or f["vehicles_involved"] in ("two", "three_or_more"))
    if not f["third_party_involved"] or f["liability_context"] == "unidentified_third_party":
        f["counterparty_identified"] = False
    else:
        f["counterparty_identified"] = pick_bool(dist["counterparty_identified"], rng)
    f["counterparty_insurance_known"] = bool(
        f["counterparty_identified"] and pick_bool(dist["counterparty_insurance_known"], rng))
    f["authority_involved"] = pick_bool(dist["authority_involved"], rng)
    if f["injury_severity"] == "severe":
        f["authority_involved"] = True
    f["event_date_known"] = pick_bool(dist["event_date_known"], rng)

    sev = f["damage_severity"]
    f["towing_needed"] = rng.random() < float(tab["towing_needed_probability"][str(sev)])
    f["vehicle_drivable"] = rng.random() < float(tab["vehicle_drivable_probability"][str(sev)])
    if f["towing_needed"]:
        f["vehicle_drivable"] = False
    stranded = (not f["vehicle_drivable"]) or f["towing_needed"]
    skey = "stranded_vehicle" if stranded else "mobile"
    f["occupants_stranded"] = rng.random() < float(tab["occupants_stranded_probability"][skey])
    f["unsafe_vehicle_condition"] = rng.random() < float(tab["unsafe_vehicle_condition_probability"][skey])

    for flag in ("glass_damage", "fire_damage", "theft_or_attempt", "weather_damage",
                 "vandalism_damage", "other_property_damage"):
        f[flag] = rng.random() < float(profile_for(cfg["conditional_facts"][flag], ev))

    f["legal_issue_type"] = sample_legal_issue(cfg, rng, f)

    f["documents_complete"] = int(pick_weighted(dist["documents_complete"], rng))
    f["urgency"] = int(pick_weighted(dist["urgency"], rng))
    if f["injury_severity"] == "severe":
        f["urgency"] = 2
    elif not f["vehicle_drivable"]:
        f["urgency"] = max(f["urgency"], 1)

    f["ambiguity_level"] = int(pick_weighted(dist["ambiguity_level"], rng))
    if controls["difficulty"] == "borderline":
        f["ambiguity_level"] = 2
    elif controls["difficulty"] == "hard":
        f["ambiguity_level"] = max(f["ambiguity_level"], 1)
    return f


def direct_compensation_ok(cfg, f) -> bool:
    req = cfg["direct_compensation"]["required"]
    for key in ("third_party_involved", "counterparty_identified", "counterparty_insurance_known"):
        if req.get(key) is not None and bool(f.get(key)) != bool(req[key]): return False
    for key, field in (("event_type_in", "event_type"),
                       ("liability_context_in", "liability_context"),
                       ("vehicles_involved_in", "vehicles_involved")):
        allowed = req.get(key)
        if allowed and f.get(field) not in allowed: return False
    return True


def human_review_ok(cfg, rng, f, controls) -> bool:
    rules = cfg["human_review_rules"]
    for field, values in rules.get("force_true_if", {}).items():
        if f.get(field) in values: return True
    p = {"easy": float(rules.get("base_probability_when_clean", 0.06)),
         "medium": float(rules.get("base_probability_when_clean", 0.06)),
         "hard": 0.22, "borderline": 0.55}.get(controls["difficulty"], 0.06)
    if f["documents_complete"] == 0: p = max(p, 0.40)
    if f["third_party_involved"] and not f["counterparty_identified"]: p = max(p, 0.35)
    if f["legal_issue_type"] != "none": p = max(p, 0.30)
    return rng.random() < p


def derive_labels(cfg, rng, f, controls) -> Dict[str, bool]:
    ev = f["event_type"]
    own_damage = f["damage_severity"] >= 1
    return {
        "rca_candidate": bool(f["third_party_involved"] and (
            f["other_property_damage"]
            or f["liability_context"] in ("insured_at_fault", "shared_fault", "third_party_at_fault")
            or ev in RCA_EVENTS)),
        "kasko_candidate": bool(own_damage and ev in KASKO_EVENTS),
        "driver_injury_candidate": bool(f["injury_reported"] and f["insured_driver_injured"]),
        "legal_protection_candidate": bool(f["legal_issue_type"] != "none"),
        "roadside_assistance_candidate": bool(
            (not f["vehicle_drivable"]) or f["towing_needed"]
            or f["occupants_stranded"] or f["unsafe_vehicle_condition"]),
        "glass_candidate": bool(f["glass_damage"]),
        "theft_candidate": bool(ev in THEFT_EVENTS or f["theft_or_attempt"]),
        "fire_candidate": bool(ev == "fire" or f["fire_damage"]),
        "natural_events_candidate": bool(ev == "weather" or f["weather_damage"]),
        "vandalism_candidate": bool(ev == "vandalism" or f["vandalism_damage"]),
        "direct_compensation_candidate": direct_compensation_ok(cfg, f),
        "needs_human_review": human_review_ok(cfg, rng, f, controls),
    }


def check_constraints(cfg, f, labels) -> List[str]:
    """Post-hoc sanity net over the logical constraints (defensive)."""
    issues: List[str] = []
    if labels["driver_injury_candidate"] and not (f["injury_reported"] and f["insured_driver_injured"]):
        issues.append("driver_injury_candidate without injured insured driver")
    if labels["glass_candidate"] and not f["glass_damage"]:
        issues.append("glass_candidate without glass_damage")
    if labels["theft_candidate"] and not (f["event_type"] in THEFT_EVENTS or f["theft_or_attempt"]):
        issues.append("theft_candidate without theft facts")
    if labels["natural_events_candidate"] and not (f["event_type"] == "weather" or f["weather_damage"]):
        issues.append("natural_events_candidate without weather facts")
    if labels["vandalism_candidate"] and not (f["event_type"] == "vandalism" or f["vandalism_damage"]):
        issues.append("vandalism_candidate without vandalism facts")
    if labels["fire_candidate"] and not (f["event_type"] == "fire" or f["fire_damage"]):
        issues.append("fire_candidate without fire facts")
    if labels["kasko_candidate"] and not (f["damage_severity"] >= 1 and f["event_type"] in KASKO_EVENTS):
        issues.append("kasko_candidate without own-vehicle damage")
    if labels["rca_candidate"] and not f["third_party_involved"]:
        issues.append("rca_candidate without third party")
    if labels["roadside_assistance_candidate"] and not (
            (not f["vehicle_drivable"]) or f["towing_needed"]
            or f["occupants_stranded"] or f["unsafe_vehicle_condition"]):
        issues.append("roadside_assistance_candidate without immobility facts")
    if labels["legal_protection_candidate"] != (f["legal_issue_type"] != "none"):
        issues.append("legal_protection_candidate inconsistent with legal_issue_type")
    if labels["direct_compensation_candidate"] and not direct_compensation_ok(cfg, f):
        issues.append("direct_compensation_candidate without prerequisites")
    if f["injured_party_role"] in NO_DRIVER_ROLES and f["insured_driver_injured"]:
        issues.append("insured_driver_injured with incompatible injured_party_role")
    if f["injury_reported"] and f["injured_party_role"] == "none":
        issues.append("injury_reported with role none")
    if not f["injury_reported"] and f["injured_party_role"] != "none":
        issues.append("no injury reported but a role is set")
    return issues


def build_blueprints(cfg, n):
    rng = random.Random(int(cfg["dataset"]["seed"]))
    ctl = cfg["controls"]
    schedules = {k: weighted_schedule(ctl[k], n, rng)
                 for k in ("event_type", "difficulty", "channel", "style")}
    sev_sched = [int(x) for x in weighted_schedule(ctl["damage_severity"], n, rng)]
    gen = cfg["generation"]
    want_rich = bool(gen.get("prefer_rich_multilabel", True))
    rich_p = float(gen.get("rich_multilabel_probability", 0.0))
    rich_min = int(gen.get("rich_multilabel_min_labels", 3))

    out = []
    for i in range(n):
        controls = {"event_type": schedules["event_type"][i],
                    "difficulty": schedules["difficulty"][i],
                    "channel": schedules["channel"][i],
                    "style": schedules["style"][i],
                    "damage_severity": sev_sched[i]}
        seed = stable_seed(cfg["dataset"]["seed"], "example", i)
        wrng = random.Random(seed)
        f = sample_world(cfg, wrng, controls)
        labels = derive_labels(cfg, wrng, f, controls)
        if want_rich and wrng.random() < rich_p:
            for _ in range(12):
                if sum(1 for v in labels.values() if v) >= rich_min: break
                f = sample_world(cfg, wrng, controls)
                labels = derive_labels(cfg, wrng, f, controls)
        targets = dict(f)
        targets.update(labels)
        out.append({"ordinal": i, "seed": seed, "controls": controls,
                    "canonical_facts": f, "targets": targets})
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# Planner / writer / critic schemas and prompts
# --------------------------------------------------------------------------

def canonical_spec(cfg) -> Dict[str, Dict[str, Any]]:
    qs = cfg["questions"]
    spec: Dict[str, Dict[str, Any]] = {}
    for field in CANONICAL_FIELD_ORDER:
        q = qs.get(field)
        if q is None:
            spec[field] = {"type": "choice", "enum": list(cfg["fact_enums"][field])}
        elif q["type"] == "choice":
            spec[field] = {"type": "choice", "enum": list(q["criteria"].keys())}
        elif q["type"] == "score":
            spec[field] = {"type": "score", "levels": len(q["criteria"]), "criteria": list(q["criteria"])}
        else:
            spec[field] = {"type": "noul"}
    return spec


def canonical_json_schema(spec) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for field, s in spec.items():
        if s["type"] == "choice": props[field] = {"type": "string", "enum": s["enum"]}
        elif s["type"] == "score": props[field] = {"type": "integer", "minimum": 0, "maximum": s["levels"]-1}
        else: props[field] = {"type": "boolean"}
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


DESCRIPTIVE_FACT_KEYS = ("event_summary", "chronology", "people_and_roles",
                         "vehicles_and_objects", "damage_and_condition", "injury_facts",
                         "documents_present", "documents_missing", "policy_context",
                         "intended_ambiguities", "contradictions", "irrelevant_distractors",
                         "must_be_explicit", "may_be_implicit")


def planner_schema(cfg):
    fact_props: Dict[str, Any] = {k: {"type": "array", "items": {"type": "string"}}
                                  for k in DESCRIPTIVE_FACT_KEYS if k != "event_summary"}
    fact_props["event_summary"] = {"type": "string"}
    return {"type": "object", "additionalProperties": False, "properties": {
        "canonical_claim": canonical_json_schema(canonical_spec(cfg)),
        "facts": {"type": "object", "additionalProperties": False,
                  "properties": fact_props, "required": list(fact_props)},
        "scenario_quality_notes": {"type": "string"}},
        "required": ["canonical_claim", "facts", "scenario_quality_notes"]}


def value_schema(q):
    if q["type"] == "choice": return {"type": "string", "enum": list(q["criteria"].keys())}
    if q["type"] == "score": return {"type": "integer", "minimum": 0, "maximum": len(q["criteria"])-1}
    return {"type": "boolean"}


def legal_values(q):
    if q["type"] == "choice": return list(q["criteria"].keys())
    if q["type"] == "score": return list(range(len(q["criteria"])))
    return [False, True]


def normalize_value(q, v):
    if q["type"] == "noul":
        if isinstance(v, bool): return v
        return str(v).strip().lower() in ("true", "1", "yes", "si", "sì")
    if q["type"] == "score":
        try: return int(v)
        except (TypeError, ValueError): return v
    return str(v)


def validate_claim(claim, requested, spec) -> List[str]:
    issues: List[str] = []
    if not isinstance(claim, Mapping):
        return ["canonical_claim is not an object"]
    for field, s in spec.items():
        if field not in claim:
            issues.append(f"{field}: missing from canonical_claim"); continue
        v = claim[field]
        if s["type"] == "choice":
            if v not in s["enum"]: issues.append(f"{field}: illegal {v!r}")
        elif s["type"] == "score":
            if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v < s["levels"]):
                issues.append(f"{field}: illegal score {v!r}")
        else:
            if not isinstance(v, bool): issues.append(f"{field}: illegal boolean {v!r}")
        if field in requested and claim[field] != requested[field]:
            issues.append(f"{field}: changed requested fact {requested[field]!r} -> {claim[field]!r}")
    return issues


WRITER_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "state": {"type": "object", "additionalProperties": False, "properties": {
        "channel": {"type": "string"}, "subject": {"type": "string"}, "message": {"type": "string"},
        "claimant_statement": {"type": "string"}, "agent_notes": {"type": "string"},
        "form": {"type": "object", "additionalProperties": False, "properties": {
            "event_date_text": {"type": "string"}, "event_location_text": {"type": "string"},
            "vehicle_status_text": {"type": "string"}, "third_party_text": {"type": "string"},
            "injury_text": {"type": "string"}, "police_or_authority_text": {"type": "string"},
            "attachments_text": {"type": "string"}},
            "required": ["event_date_text", "event_location_text", "vehicle_status_text",
                         "third_party_text", "injury_text", "police_or_authority_text",
                         "attachments_text"]}},
        "required": ["channel", "subject", "message", "claimant_statement", "agent_notes", "form"]},
    "writer_notes": {"type": "string"}}, "required": ["state", "writer_notes"]}

CRITIC_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "accept": {"type": "boolean"},
    "realism_score": {"type": "integer", "minimum": 0, "maximum": 100},
    "label_alignment_score": {"type": "integer", "minimum": 0, "maximum": 100},
    "information_fidelity_score": {"type": "integer", "minimum": 0, "maximum": 100},
    "style_quality_score": {"type": "integer", "minimum": 0, "maximum": 100},
    "label_leakage": {"type": "boolean"},
    "material_facts_invented": {"type": "boolean"},
    "material_facts_omitted": {"type": "boolean"},
    "issues": {"type": "array", "items": {"type": "string"}},
    "repair_instructions": {"type": "array", "items": {"type": "string"}},
    "decision_checks": {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"decision_id": {"type": "string"},
                       "supported_by_state": {"type": "boolean"},
                       "comment": {"type": "string"}},
        "required": ["decision_id", "supported_by_state", "comment"]}},
    "required": ["accept", "realism_score", "label_alignment_score",
                 "information_fidelity_score", "style_quality_score", "label_leakage",
                 "material_facts_invented", "material_facts_omitted", "issues",
                 "repair_instructions", "decision_checks"]}}


class OpenRouter:
    def __init__(self, cfg):
        self.cfg = cfg; self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key: raise SystemExit("Set OPENROUTER_API_KEY")

    def call(self, model_cfg, system, user, name, schema, seed):
        base = {"model": model_cfg["model"],
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}], "seed": seed}
        for k in ("temperature", "max_tokens"):
            if model_cfg.get(k) is not None: base[k] = model_cfg[k]
        for k, v in (model_cfg.get("extra") or {}).items():
            if k not in {"model", "messages", "response_format"}: base[k] = v
        formats = [{"type": "json_schema", "json_schema": {"name": name[:64], "strict": True, "schema": schema}}]
        if self.cfg.get("structured_output_fallback", True): formats.append({"type": "json_object"})
        if self.cfg.get("structured_output_first", False):
            formats = formats[1:] + formats[:1]
        last = None
        for fmt in formats:
            for attempt in range(int(self.cfg.get("max_http_retries", 6))):
                payload = dict(base); payload["response_format"] = fmt
                headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                           "User-Agent": f"laya-dataset/{VERSION}"}
                if self.cfg.get("http_referer"): headers["HTTP-Referer"] = self.cfg["http_referer"]
                if self.cfg.get("x_title"): headers["X-Title"] = self.cfg["x_title"]
                req = urllib.request.Request(self.cfg.get("api_url", OPENROUTER_URL),
                                             data=json.dumps(payload).encode(),
                                             headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=int(self.cfg.get("timeout_seconds", 180))) as r:
                        env = json.loads(r.read())
                    msg = env["choices"][0]["message"].get("content", "")
                    if isinstance(msg, list):
                        msg = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in msg)
                    msg = str(msg).strip()
                    msg = re.sub(r"^```(?:json)?\s*|\s*```$", "", msg, flags=re.I)
                    try: data = json.loads(msg)
                    except json.JSONDecodeError:
                        a, b = msg.find("{"), msg.rfind("}"); data = json.loads(msg[a:b+1])
                    if not isinstance(data, dict): raise ValueError("response is not object")
                    return data, env.get("model"), env.get("usage") or {}
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
                    last = RuntimeError(f"HTTP {e.code}: {body[:1000]}")
                    if e.code in (400, 404, 422): break
                    if e.code not in (408, 409, 429, 500, 502, 503, 504): raise last
                    ra = e.headers.get("Retry-After")
                    delay = float(ra) if ra and ra.replace('.', '', 1).isdigit() else float(self.cfg.get("retry_base_seconds", 2))*(2**attempt)
                    time.sleep(min(delay, 60))
                except Exception as e:
                    last = e
                    time.sleep(min(float(self.cfg.get("retry_base_seconds", 2))*(2**attempt), 60))
        raise RuntimeError(f"OpenRouter failed for {model_cfg['model']}: {last}")


def plan_prompt(cfg, bp):
    d = cfg["dataset"]
    spec = canonical_spec(cfg)
    return f"""Create ONE canonical synthetic claim.
Context: language={d['language']}, locale={d['locale']}, jurisdiction={d['jurisdiction']}, product={d['product']}.
Controls (how the case must be written later, do not encode them as facts):
{pretty({k: v for k, v in bp['controls'].items() if k in ('difficulty', 'channel', 'style')})}

FIXED canonical claim (authoritative, reproduce EXACTLY, field by field):
{pretty(bp['canonical_facts'])}

Allowed values per field:
{pretty(spec)}

Return:
- canonical_claim: the fixed canonical claim, echoed exactly;
- facts: rich descriptive material (chronology, roles, objects, damage, injury, documents, policy context, intended ambiguities, contradictions, distractors) so that a writer can realize this exact case without seeing any internal taxonomy or decision name. Declare explicitly in must_be_explicit the facts that must survive into the final text, and in may_be_implicit what may stay implicit.
- scenario_quality_notes: how the case stays defensible while being {bp['controls']['difficulty']}."""


def writer_prompt(cfg, bp, facts, prior=None, repair=None):
    d = cfg["dataset"]; extra = ""
    if prior is not None:
        extra = (f"\nPrevious state:\n{pretty(prior)}\nReviewer repair instructions:\n"
                 f"{pretty(repair or [])}\nRewrite coherently from the canonical facts.")
    return f"""Write ONE fictional insurance record.
Language={d['language']} locale={d['locale']} jurisdiction={d['jurisdiction']} product={d['product']}.
Channel={bp['controls']['channel']} style={bp['controls']['style']} difficulty={bp['controls']['difficulty']}.
Canonical facts of this claim:
{pretty(bp['canonical_facts'])}

Descriptive material to realize:
{pretty(facts)}{extra}

state.channel must equal {bp['controls']['channel']!r}. Use empty strings for inapplicable form/agent fields.
Realize every value of the canonical facts and every must_be_explicit item, keep the declared missing information missing, and never name a coverage, guarantee, candidate or decision."""


def critic_prompt(cfg, bp, facts, state):
    qs = cfg["questions"]
    labels = {k: bp["targets"][k] for k in LABEL_IDS}
    facts_only = {k: bp["targets"][k] for k in CANONICAL_FIELD_ORDER}
    return f"""Audit one candidate claim record.

Canonical facts (ground truth world, authoritative):
{pretty(facts_only)}

Descriptive material:
{pretty(facts)}

Decision schema (question types):
{pretty({k: qs[k] for k in LABEL_IDS})}

Intended decisions:
{pretty(labels)}

Generated state (the only thing a Laya model will see):
{pretty(state)}

For EVERY decision id listed below, return one decision_checks entry with supported_by_state=true only when a careful human can defend that exact value from the state ALONE:
{pretty(list(LABEL_IDS))}

Additionally verify: injury role coherent with text; liability context coherent with the described dynamics; vehicle_drivable and towing_needed coherent with the described damage; legal protection coherent with a real dispute; direct compensation prerequisites coherent; no label leakage (no coverage/guarantee/candidate/decision wording); no material fact invented; no material fact omitted."""


def critic_ok(review, g, required_ids):
    if not review.get("accept"): return False
    checks = review.get("decision_checks") or []
    seen = {c.get("decision_id") for c in checks}
    if not set(required_ids) <= seen: return False
    for k, minimum in (("realism_score", g["min_realism_score"]),
                       ("label_alignment_score", g["min_label_alignment_score"]),
                       ("information_fidelity_score", g["min_information_fidelity_score"]),
                       ("style_quality_score", g["min_style_quality_score"])):
        if int(review.get(k, 0)) < int(minimum): return False
    if review.get("material_facts_invented") or review.get("material_facts_omitted"): return False
    if g.get("reject_label_leakage", True) and review.get("label_leakage"): return False
    return all(c.get("supported_by_state") for c in checks)


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.orc = OpenRouter(cfg["openrouter"])
        self.ps = planner_schema(cfg)
        self.spec = canonical_spec(cfg)

    def generate(self, bp):
        c = self.cfg; g = c["generation"]; m = c["models"]
        rng = random.Random(bp["seed"]); rejected = []

        def jitter():
            lo, hi = g.get("request_jitter_seconds", [0, 0])
            if float(hi) > 0: time.sleep(rng.uniform(float(lo), float(hi)))

        for pa in range(int(g["max_plan_attempts"])):
            jitter()
            plan, pm, _ = self.orc.call(m["planner"], PLANNER_SYSTEM, plan_prompt(c, bp),
                                        "laya_claim_plan", self.ps, stable_seed(bp["seed"], "plan", pa))
            claim = plan.get("canonical_claim", {})
            issues = validate_claim(claim, bp["canonical_facts"], self.spec)
            if issues:
                rejected.append({"stage": "planner", "issues": issues}); continue
            facts = plan.get("facts", {})
            prior = repair = None
            for wa in range(int(g["max_writer_repairs"]) + 1):
                jitter()
                wr, wm, _ = self.orc.call(m["writer"], WRITER_SYSTEM, writer_prompt(c, bp, facts, prior, repair),
                                          "laya_claim_record", WRITER_SCHEMA, stable_seed(bp["seed"], "writer", pa, wa))
                state = wr.get("state", {})
                leaks = leakage_scan(state)
                if leaks:
                    rejected.append({"stage": "leakage_guard", "tokens": leaks})
                    prior, repair = state, ["Remove every internal label/coverage/decision wording."]
                    continue
                jitter()
                rev, cm, _ = self.orc.call(m["critic"], CRITIC_SYSTEM, critic_prompt(c, bp, facts, state),
                                           "laya_claim_review", CRITIC_SCHEMA, stable_seed(bp["seed"], "critic", pa, wa))
                if critic_ok(rev, g, LABEL_IDS):
                    rid = hashlib.sha256(f"{c['dataset']['name']}|{bp['seed']}|{canonical(state)}".encode()).hexdigest()[:20]
                    rec = {"id": f"syn-{rid}", "schema_id": c["dataset"]["schema_id"],
                           "source": "synthetic_openrouter", "state": state,
                           "targets": bp["targets"],
                           "metadata": {"synthetic": True, "controls": bp["controls"], "seed": bp["seed"],
                                        "planner_model": pm or m["planner"]["model"],
                                        "writer_model": wm or m["writer"]["model"],
                                        "critic_model": cm or m["critic"]["model"],
                                        "critic": rev, "generated_at": now(),
                                        "generator_version": VERSION,
                                        "label_cardinality": sum(1 for k in LABEL_IDS if bp["targets"][k])}}
                    if c["dataset"].get("include_questions_in_records", True): rec["questions"] = c["questions"]
                    if c["dataset"].get("keep_generation_facts", True):
                        rec["generation"] = {"canonical_facts": bp["canonical_facts"],
                                             "descriptive_facts": facts,
                                             "scenario_quality_notes": plan.get("scenario_quality_notes", "")}
                    return rec, rejected
                rejected.append({"stage": "critic", "review": rev})
                prior, repair = state, rev.get("repair_instructions", [])
        brief = []
        for r in rejected[-6:]:
            brief.append({k: v for k, v in r.items() if k != "blueprint"})
        raise RuntimeError("quality gate exhausted: " + json.dumps(brief, ensure_ascii=False)[:1600])


# --------------------------------------------------------------------------
# Dedupe / split / report
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"\w+", re.UNICODE)


def state_text(x):
    if isinstance(x, str): return x
    if isinstance(x, Mapping): return " ".join(state_text(v) for v in x.values())
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)): return " ".join(state_text(v) for v in x)
    return ""


def leakage_scan(state) -> List[str]:
    tokens = set(WORD_RE.findall(unicodedata.normalize("NFKC", state_text(state)).lower()))
    return sorted(tokens & LEAK_TOKENS)


def simhash(text):
    toks = WORD_RE.findall(unicodedata.normalize("NFKC", text).lower())
    shingles = toks if len(toks) < 3 else [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    vec = [0]*64
    for s in shingles:
        h = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
        for b in range(64): vec[b] += 1 if (h >> b) & 1 else -1
    return sum((1 << b) for b, v in enumerate(vec) if v >= 0)


class LSH:
    def __init__(self, d): self.d = d; self.h = []; self.b = defaultdict(list)
    def keys(self, h):
        for i in range(4): yield (i, (h >> (16*i)) & 0xffff)
    def near(self, h):
        cand = set()
        for k in self.keys(h): cand.update(self.b[k])
        for i in cand:
            if (h ^ self.h[i]).bit_count() <= self.d: return i
        return None
    def add(self, h):
        i = len(self.h); self.h.append(h)
        for k in self.keys(h): self.b[k].append(i)


def dedupe(rows, maxdist):
    out = []; dup = []; exact = set(); lsh = LSH(maxdist)
    for r in rows:
        text = " ".join(WORD_RE.findall(unicodedata.normalize("NFKC", state_text(r["state"])).lower()))
        dg = hashlib.sha256(text.encode()).hexdigest(); sh = simhash(text)
        if dg in exact: dup.append((r, "exact")); continue
        j = lsh.near(sh)
        if j is not None: dup.append((r, f"near:{out[j]['id']}")); continue
        exact.add(dg); lsh.add(sh); out.append(r)
    return out, dup


def split_rows(rows, cfg):
    sc = cfg["dataset"]["splits"]; names = list(sc); ws = [float(sc[n]) for n in names]
    s = sum(ws); ws = [x/s for x in ws]
    key = cfg["dataset"]["stratify_on"]
    groups = defaultdict(list)
    for r in rows: groups[str(r["targets"].get(key))].append(r)
    rng = random.Random(stable_seed(cfg["dataset"]["seed"], "splits")); out = {n: [] for n in names}
    for items in groups.values():
        rng.shuffle(items); n = len(items); raw = [n*w for w in ws]
        cnt = [int(math.floor(x)) for x in raw]; rem = n - sum(cnt)
        order = sorted(range(len(names)), key=lambda i: (raw[i]-cnt[i], rng.random()), reverse=True)
        for i in order[:rem]: cnt[i] += 1
        p = 0
        for name, c in zip(names, cnt):
            chunk = items[p:p+c]; p += c
            for r in chunk: r["split"] = name
            out[name] += chunk
    for x in out.values(): rng.shuffle(x)
    return out


def report(rows, cfg):
    qs = cfg["questions"]
    dist = {qid: dict(Counter(str(r["targets"].get(qid)) for r in rows)) for qid in qs}
    for k in ("event_type", "difficulty", "channel", "style", "damage_severity"):
        dist["control:"+k] = dict(Counter(str(r["metadata"]["controls"].get(k)) for r in rows))

    total = max(1, len(rows))
    combos = {}
    for combo in cfg.get("report", {}).get("multi_label_combos", []):
        n = sum(1 for r in rows if all(r["targets"].get(c) for c in combo))
        combos[" + ".join(combo)] = {"count": n, "share": round(n/total, 4)}
    cardinality = dict(Counter(str(sum(1 for k in LABEL_IDS if r["targets"].get(k))) for r in rows))

    by_event = {}
    for et in sorted({str(r["targets"].get("event_type")) for r in rows}):
        sub = [r for r in rows if str(r["targets"].get("event_type")) == et]
        by_event[et] = {"count": len(sub),
                        "label_prevalence": {k: round(sum(1 for r in sub if r["targets"].get(k))/max(1, len(sub)), 4)
                                             for k in LABEL_IDS}}

    scores = {}
    for k in ("realism_score", "label_alignment_score", "information_fidelity_score", "style_quality_score"):
        v = [float(r["metadata"]["critic"].get(k, 0)) for r in rows]
        scores[k] = {"min": min(v), "mean": round(sum(v)/len(v), 2), "max": max(v)} if v else {}

    return {"count": len(rows), "dataset": cfg["dataset"]["name"], "schema_id": cfg["dataset"]["schema_id"],
            "generator_version": VERSION, "distributions": dist,
            "label_cardinality": cardinality, "multi_label_combos": combos,
            "label_prevalence_by_event_type": by_event, "critic_scores": scores,
            "target_probabilities": False,
            "note": ("Synthetic benchmark only: targets are discrete/ordinal ground truth. "
                     "Per-option probabilities are produced by Laya at inference time and are not part of this dataset. "
                     "Validate production performance on held-out real customer data.")}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_dry(cfg, n, out):
    blue = build_blueprints(cfg, n)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"ordinal": b["ordinal"], "controls": b["controls"],
             "canonical_facts": b["canonical_facts"], "targets": b["targets"]} for b in blue]
    write_jsonl(out / "blueprints.dry.jsonl", rows)
    problems = []
    for b in blue:
        problems += [f"{b['ordinal']}: {m}" for m in check_constraints(cfg, b["canonical_facts"], b["targets"])]
    label_counts = {k: sum(1 for b in blue if b["targets"][k]) for k in LABEL_IDS}
    cardinality = dict(Counter(str(sum(1 for k in LABEL_IDS if b["targets"][k])) for b in blue))
    print(json.dumps({"blueprints": len(blue), "constraint_issues": problems[:50],
                      "constraint_issue_count": len(problems),
                      "label_counts": label_counts, "label_cardinality": cardinality,
                      "event_types": dict(Counter(b["controls"]["event_type"] for b in blue))},
                     indent=2, ensure_ascii=False))
    return blue


def main():
    ap = argparse.ArgumentParser(description="Laya synthetic insurance claims generator")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("data/insurance_synthetic"))
    ap.add_argument("--count", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--init-config", type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="sample canonical claims and validate constraints without calling OpenRouter")
    ap.add_argument("--validate-config", action="store_true",
                    help="validate the configuration and exit")
    a = ap.parse_args()

    if a.init_config:
        a.init_config.parent.mkdir(parents=True, exist_ok=True)
        dump_json(a.init_config, DEFAULT_CONFIG); print(a.init_config); return

    cfg = load_config(a.config)
    issues = validate_config(cfg)
    if issues:
        for i in issues: print("config error:", i, file=sys.stderr)
        raise SystemExit(f"{len(issues)} configuration error(s)")
    if a.validate_config:
        print(json.dumps({"ok": True, "questions": len(cfg["questions"]),
                          "event_types": len(cfg["controls"]["event_type"]),
                          "labels": list(LABEL_IDS)}, indent=2)); return

    n = a.count or int(cfg["dataset"]["count"])
    out = a.output_dir
    if a.dry_run:
        run_dry(cfg, n, out); return

    workers = a.workers or int(cfg["generation"]["workers"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not a.overwrite:
        raise SystemExit("output-dir is not empty; use --overwrite or another directory")
    if a.overwrite:
        for p in out.iterdir():
            if p.is_file(): p.unlink()
    dump_json(out/"effective_config.json", cfg)
    dump_json(out/"questions.json", {"schema_id": cfg["dataset"]["schema_id"], "questions": cfg["questions"]})

    rejected = out/"rejected.work.jsonl"; lock = threading.Lock()
    gen = Generator(cfg); blue = build_blueprints(cfg, n)

    def job(bp):
        try: return bp, *gen.generate(bp), None
        except Exception as e: return bp, None, [], f"{type(e).__name__}: {e}"

    def drain(futs, tag):
        out = []
        for i, f in enumerate(cf.as_completed(futs), 1):
            bp, rec, rejs, err = f.result()
            for x in rejs: append_jsonl(rejected, {"blueprint": bp, "detail": x, "at": now()}, lock)
            if rec is None:
                failed.append(bp)
                append_jsonl(rejected, {"blueprint": bp, "error": err, "at": now()}, lock)
                print(f"[{tag} {i}] reject {err}", file=sys.stderr)
            else:
                out.append(rec)
                print(f"[{tag} {i}] {rec['id']} {rec['targets']['event_type']} "
                      f"labels={rec['metadata']['label_cardinality']}")
        return out

    rows = []; failed = []
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = drain([pool.submit(job, b) for b in blue], f"1/{n}")
    if not rows:
        raise SystemExit(f"No case passed the quality gate out of {n}; inspect {rejected}")

    rows, dups = dedupe(rows, int(cfg["generation"]["near_duplicate_hamming_distance"]))
    for r, why in dups: append_jsonl(rejected, {"stage": "dedupe", "record_id": r["id"], "reason": why}, lock)
    needs = failed + [next((b for b in blue if b["seed"] == r["metadata"]["seed"]), blue[0]) for r, _ in dups]
    roundno = 0
    while len(rows) < n and roundno < int(cfg["generation"]["replacement_rounds"]):
        roundno += 1; need = n - len(rows); jobs = []
        for i in range(need):
            b = copy.deepcopy((needs or blue)[i % len(needs or blue)])
            b["seed"] = stable_seed(cfg["dataset"]["seed"], "replacement", roundno, i, b["seed"])
            jobs.append(b)
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            new = drain([pool.submit(job, b) for b in jobs], f"r{roundno}")
        rows, dups = dedupe(rows+new, int(cfg["generation"]["near_duplicate_hamming_distance"]))
    if len(rows) < n:
        raise SystemExit(f"Only {len(rows)}/{n} unique accepted cases; inspect {rejected}")

    rows = rows[:n]; splits = split_rows(rows, cfg)
    write_jsonl(out/"all.jsonl", rows)
    for name, data in splits.items():
        write_jsonl(out/f"{name}.jsonl", data)
        write_jsonl(out/f"{name}.laya.jsonl", (
            {"id": r["id"], "schema_id": r["schema_id"], "state": r["state"],
             "questions": cfg["questions"], "targets": r["targets"],
             "generation": {"canonical_facts": r.get("generation", {}).get("canonical_facts", {})},
             "split": name} for r in data))
    rep = report(rows, cfg); rep["splits"] = {k: len(v) for k, v in splits.items()}
    dump_json(out/"report.json", rep)
    dump_json(out/"generation_manifest.json", {
        "version": VERSION, "completed_at": now(), "count": len(rows),
        "config_sha256": hashlib.sha256(canonical(cfg).encode()).hexdigest(),
        "models": cfg["models"], "labels": list(LABEL_IDS),
        "canonical_fields": list(CANONICAL_FIELD_ORDER)})
    print("DONE", out); print(json.dumps(rep["splits"], indent=2))


if __name__ == "__main__":
    main()
