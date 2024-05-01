import { useState, useEffect } from 'react'

import Title from './Title'
import Prompt from './Prompt'

const CreatePrompt = () => {
    const [prompts, setPrompts] = useState([])
    const [error, setError] = useState('')
    const [newPrompt, setNewPrompt] = useState('')
    const [newPromptType, setNewPromptType] = useState('')
    const [categories, setCategories] = useState([])
    const [currentCategory, setCurrentCategory] = useState('')
    const [subcategories, setSubcategories] = useState([])
    const [currentSubcategory, setCurrentSubcategory] = useState('')

    // const URL = 'http://localhost:8000'
    const URL = 'https://train.synapse.com.np'

    const getSubcategories = async () => {
        if (currentCategory === '') {
            return
        }
        const res = await fetch(URL + '/subcategories/',
            {
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json',

                },
                method: 'POST',
                body: JSON.stringify({ category: currentCategory })
            }
        )
        const data = await res.json()
        setSubcategories(data)
    }

    useEffect(() => {
        getSubcategories()
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [currentCategory])

    const createPrompt = async () => {
        // send a post request with the current prompt to the server
        // on successful request call the getPrompts function
        if (newPrompt === '' ) {
            setError('Prompt cannot be empty.')
            return
        }
        if (newPromptType === '') {
            setError('Prompt type cannot be empty.')
            return
        }

        if (currentCategory === '' || currentSubcategory === '') {
            setError('Category and subcategory cannot be empty.')
            return
        }

        const res = await fetch(URL + '/prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify({
                prompt: newPrompt,
                type: newPromptType,
                category: currentCategory,
                subcategory: currentSubcategory
            })
        })

        if (res.status === 201 || res.status === 200) {
            getPrompts()
            setError('Success!')
            setNewPrompt('')
            setNewPromptType('')
        } else {
            setError('Failed to create prompt.')
        }

        // clear the newPrompt state
    }

    const getPrompts = async () => {
        // send a get request to the server to get all the prompts
        // on successful request set the prompts state with the response
        const res = await fetch(URL + '/prompts',
            {
                headers: {
                    'Access-Control-Allow-Origin': '*'
                }
            }
        )
        const data = await res.json()
        setPrompts(data)

        // setPrompts([])
    }

    const getCategories = async () => {
        const res = await fetch(URL + '/categories',
            {
                headers: {
                    'Access-Control-Allow-Origin': '*'
                }
            }
        )
        const data = await res.json()
        setCategories(data)
    }

    // on page load get all prompts
    useEffect(() => {
        getPrompts()
        console.log('fetching categories')
        getCategories()
    }, [])

    useEffect(() => {
        setError('')
    }, [currentCategory, currentSubcategory])

    return (
        <section className='w-full lg:w-[50%] flex flex-col p-4 md:p-10 h-screen min-h-screen max-h-screen'>
            {/* Left section to create and view prompts */}
            {/* Title of the left side */}
            <Title text='Create Prompt' />

            {/* Form to create new prompt */}
            <div className='w-full flex flex-col p-4 gap-2'>
                <label className='font-extrabold text-slate-900'>Enter a new prompt</label>
                <input type='text' placeholder='Prompt...' className='border-2 border-slate-700 rounded p-1'
                    value={newPrompt} onChange={(e) => setNewPrompt(e.target.value)}
                />
                {/* two buttons with values: suppress, boost. The selected value is highlighted, and the corresponding set state is called
                    initial value is none
                */}
                <label className='font-extrabold text-slate-900'>Select Prompt Type</label>
                <div className='flex flex-row gap-2'>
                    <button className={`border-2 border-slate-700 rounded p-1 ${newPromptType === 'suppress' ? 'bg-slate-700 text-white' : ''}`} onClick={() => setNewPromptType('suppress')}>Suppress</button>
                    <button className={`border-2 border-slate-700 rounded p-1 ${newPromptType === 'boost' ? 'bg-slate-700 text-white' : ''}`} onClick={() => setNewPromptType('boost')}>Boost</button>
                </div>

                {/* A dropdown that displays categories */}
                <label className='font-extrabold text-slate-900'>Select Category</label>
                <select className='border-2 border-slate-700 rounded p-1' onChange={(e) => setCurrentCategory(e.target.value)}>
                    <option value=''>Select Category</option>
                    {categories.map((category, index) => (
                        <option key={index} value={category}>{category}</option>
                    ))}
                </select>

                <label className='font-extrabold text-slate-900'>Select Subcategory</label>
                <select className='border-2 border-slate-700 rounded p-1' onChange={(e) => setCurrentSubcategory(e.target.value)}>
                    <option value=''>Select Subcategory</option>
                    {subcategories && subcategories.map((subcategory, index) => (
                        <option key={index} value={subcategory}>{subcategory}</option>
                    ))}
                </select>


                <button className='border-2 border-slate-700 rounded p-1' onClick={createPrompt}>Create Prompt</button>
                <hr className='w-full border-1 mt-4 border-slate-950' />
            </div>
            <p className='w-full p-2 pl-4 text-slate-800 text-bold text-center'>{error}</p>

            <Title text='Prompts' />
            {/* List of all prompts */}
            <div className='overflow-auto w-full h-full p-4 flex flex-col gap-2'>
                {/* check the len of prompts, if len of prompts is not 0, iterate through it and render <Prompt id={},text={}/> */}
                {prompts.length !== 0 ? prompts.map((prompt, index) => (
                    <Prompt key={index} id={prompt.id} text={prompt.prompt} type={prompt.type} />
                )) : 'No prompts available'}
            </div>
        </section>
    )
}


export default CreatePrompt
