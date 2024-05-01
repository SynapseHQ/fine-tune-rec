/* eslint-disable react/prop-types */
import Prompt from './Prompt'
import { useState } from 'react'

const Label = ({ combination, setIndex }) => {
    const [label, setLabel] = useState(0.5)
    const [error, setError] = useState('')

    // const URL = 'http://localhost:8000'
    const URL = 'https://train.synapse.com.np'
    const submitLabel = async () => {
        // send a post request to the server with the current combination and the label
        //   setIndex((prev) => prev + 1)
        console.log("submit label")
        const response = await fetch(URL + '/label', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify({
                post: combination.post.id,
                prompt: combination.prompt.id,
                label
            })
        })

        if (response.status === 201 || response.status === 200) {
            setIndex((prev) => prev + 1)
            setError('Success!')
        } else {
            setError('Failed to create data label.')
        }
    }

    if (!combination) {
        return (
            <div className='w-full p-4'>
                <p>Loading...</p>
            </div>
        )
    }

    return (
        <div className='p-4 flex flex-col gap-4'>
            <div className='h-[25vh] lg:[30vh] overflow-auto'>
                <div className='flex justify-start gap-2'>
                    <p className='font-bold text-base text-slate-950'>{combination.post.category} | </p>
                    <p className='font-bold text-base text-slate-700'>{combination.post.subcategory}</p>
                </div>
                <hr className='w-full border-2 border-slate-900' />
                <p className='font-extrabold text-lg text-slate-900 pt-4'>{combination.post.title}</p>
                <p className='text-slate-800 overflow-'>{combination.post.abstract}</p>
            </div>
            {/* a elegant section breaker line */}
            <hr className='w-full border-1 border-slate-950' />
            {/* Label */}
            <p className='font-extrabold text-lg text-slate-900'>Prompt</p>
            <Prompt
                id={combination.prompt.id}
                text={combination.prompt.prompt}
                type={combination.prompt.type}
                category={combination.prompt.category}
                subcategory={combination.prompt.subcategory}
            />
            {/* buttons ranging from 0 to 1 in increments of 0.1
            the selected button is the label value
            the selected button should be highlighted
        */}
            <div className='flex md:flex-row justify-evenly w-full flex-wrap'>
                {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((value, index) => (
                    <button key={index} onClick={() => setLabel(value)} className={`w-[25%] lg:w-[10%] border-2 m-1 border-slate-900 p-1 rounded font-bold text-xl ${label === value ? 'bg-slate-900 text-white' : ''}`}>{value}</button>
                ))}
            </div>
            {/* button to skip */}
            <button onClick={() => setIndex((prev) => prev + 1)} className='border-2 border-slate-900 p-1 rounded font-bold text-xl'>Skip</button>
            {/* button to submit */}
            <button onClick={submitLabel} className='border-2 border-slate-900 p-1 rounded font-bold text-xl '>Submit</button>
            <p className='text-slate-900 w-full text-center'>{error}</p>
        </div>
    )
}


export default Label
