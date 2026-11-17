# Renders chat templates with jinja2 configured as transformers' apply_chat_template does.
import json, sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols
out = sys.argv[1]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(m): raise Exception(m)
env.globals["raise_exception"] = raise_exception
T = {}
T["chatml"] = r"""{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
T["llama3"] = r"""{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
T["mistral"] = r"""{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}

{{- bos_token }}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {%- if loop.first and system_message is defined %}
            {{- ' [INST] ' + system_message + '\n\n' + message['content'] + ' [/INST]' }}
        {%- else %}
            {{- ' [INST] ' + message['content'] + ' [/INST]' }}
        {%- endif %}
    {%- elif message['role'] == 'assistant' %}
        {{- ' ' + message['content'] + eos_token}}
    {%- else %}
        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}
    {%- endif %}
{%- endfor %}
"""
T["qwen-tools"] = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""
T["gemma-ns"] = r"""{{ bos_token }}{% set ns = namespace(count=0, sys='') %}{% if messages[0]['role'] == 'system' %}{% set ns.sys = messages[0]['content'] | trim + '\n\n' %}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{% set ns.count = ns.count + 1 %}{{ '<start_of_turn>' + role + '\n' + (ns.sys if loop.first else '') + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}{{ ns.count }}"""
T["misc"] = r"""{# a comment #}
{% macro say(x, y='!') -%}
  <{{ x | upper }}{{ y }}>
{%- endmacro %}
{% for k, v in {'b': 2, 'a': 1}.items() %}{{ k }}={{ v }};{% endfor %}
{{ say('hi') }} {{ say('yo', y='?') }}
{{ [3,1,2] | length }} {{ ['x','y','z'] | join('-') }} {{ 'abc'[::-1] }} {{ 'a,b,c'.split(',')[1:] }}
{{ 7 // 2 }} {{ 7 % 3 }} {{ 2 * 3 + 1 }} {{ 'x' ~ 1 ~ none }} {{ 1.5 + 1 }}
{% if undefined_thing is not defined and 'b' in 'abc' and 3 not in [1, 2] %}ok{% endif %}
{%- for m in messages if m.role != 'system' %}
  [{{ loop.index }}/{{ loop.length }}{{ ' last' if loop.last }}] {{ m.content.strip() | replace('o', '0') }}
{%- else %}
none
{%- endfor %}
{{ messages | selectattr('role', 'equalto', 'user') | map(attribute='content') | list | tojson }}
{{ {'k': [1, 'two', none, true]} | tojson(indent=2) }}
{{ "  pad  " | trim | capitalize }}|{{ "Hello World".startswith("Hell") }}|{{ range(3) | list }}
"""
msgs = {
 "basic": [{"role":"system","content":"You are terse."},{"role":"user","content":"  Hi there  "},{"role":"assistant","content":"Hello."},{"role":"user","content":"Bye ünïcode 🙂"}],
 "nosys": [{"role":"user","content":"Q1"},{"role":"assistant","content":"A1"},{"role":"user","content":"Q2"}],
 "tools": [{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris","unit":"c"}}}]},{"role":"tool","content":"{\"temp\": 21}"},{"role":"assistant","content":"21 degrees."}],
}
tools = [{"type":"function","function":{"name":"get_weather","description":"Get the weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
cases = []
for name, src in T.items():
    open(f"{out}/{name}.jinja", "w").write(src)
    for mname, m in msgs.items():
        for gen in (False, True):
            for use_tools in ((False, True) if name == "qwen-tools" else (False,)):
                ctx = dict(messages=m, add_generation_prompt=gen, bos_token="<s>", eos_token="</s>")
                if use_tools: ctx["tools"] = tools
                try:
                    r = {"text": env.from_string(src).render(**ctx)}
                except Exception as e:
                    r = {"error": str(e)}
                cases.append(dict(template=name, messages=mname, add_generation_prompt=gen, tools=use_tools, **r))
json.dump({"messages": msgs, "tools": tools, "cases": cases}, open(f"{out}/expected.json", "w"), ensure_ascii=False, indent=1)
print(len(cases), sum('error' in c for c in cases))
