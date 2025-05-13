from uuid import uuid4
from .common.client import A2ACardResolver, A2AClient
from .common.client.types import TaskState


async def perform_action(
    userPrompt,
    agent,
    session=0,
    history=False,
):
    card_resolver = A2ACardResolver(agent)
    card = card_resolver.get_agent_card()

    print('======= Agent Card ========')
    print(card.model_dump_json(exclude_none=True))

    client = A2AClient(agent_card=card)
    if session == 0:
        sessionId = uuid4().hex
    else:
        sessionId = session

    # streaming = card.capabilities.streaming
    streaming = False

    # Call completeTask just once with a single taskId
    taskId = uuid4().hex
    print('=========  starting a new task ======== ')
    result = await completeTask(
        client,
        streaming,
        taskId,
        sessionId,
        userPrompt
    )

    if history and result:
        print('========= history ======== ')
        task_response = await client.get_task(
            {'id': taskId, 'historyLength': 10}
        )
        print(
            task_response.model_dump_json(
                include={'result': {'history': False}}
            )
        )

    return result


async def completeTask(
    client: A2AClient,
    streaming,
    taskId,
    sessionId,
    userPrompt
):
    prompt = userPrompt
    message = {
        'role': 'user',
        'parts': [
            {
                'type': 'text',
                'text': prompt,
            }
        ],
    }

    payload = {
        'id': taskId,
        'sessionId': sessionId,
        'acceptedOutputModes': ['text'],
        'message': message,
    }

    taskResult = None
    if streaming:
        response_stream = client.send_task_streaming(payload)
        async for result in response_stream:
            print(
                f'stream event => {result.model_dump_json(exclude_none=False)}'
            )
        taskResult = await client.get_task({'id': taskId})
    else:
        taskResult = await client.send_task(payload)
        print(f'event msg =>\n{taskResult.model_dump_json(exclude_none=True)}')

    ## if the result is that more input is required, loop again.
    state = TaskState(taskResult.result.status.state)
    if state.name == TaskState.INPUT_REQUIRED.name:
        return await completeTask(
            client,
            streaming,
            taskId,
            sessionId,
            userPrompt
        )
    ## task is complete
    return taskResult.result.artifacts[0].parts[0].text