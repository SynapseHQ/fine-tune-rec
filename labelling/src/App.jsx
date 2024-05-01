/* eslint-disable react/prop-types */
import { useState, useEffect } from 'react'

import './index.css'
import Title from './components/Title'
import Label from './components/Label'
import CreatePrompt from './components/CreatePrompt'

// load api url from .env

const AssignLabel = () => {
  const [labelCount, setLabelCount] = useState(0)
  const [combinations, setCombinations] = useState([])
  const [index, setIndex] = useState(0)
  // const URL = 'http://localhost:8000'
  const URL = 'https://train.synapse.com.np'
  const getCombinations = async () => {
    const response = await fetch(URL + '/train', {
      headers: {
        'Access-Control-Allow-Origin': '*'
      }
    })
    const data = await response.json()
    setCombinations(data.combinations)
  }

  const getLabelCount = async () => {
    const response = await fetch(URL + '/dataset_size')
    const data = await response.json()
    setLabelCount(data.size)
  }

  useEffect(() => {
    getCombinations()
    getLabelCount()
  }, [])

  useEffect(() => {
    if (index === combinations.length) {
      getCombinations()
      setIndex(0)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index])
  return (
    <>
      <section className='w-full lg:w-[50%] flex flex-col p-4 md:p-10 h-screen min-h-screen max-h-screen'>
        {/* Left section to create and view prompts */}
        {/* Title of the left side */}
        <div className='flex justify-between items-center font-bold pl-4 pr-4'>
          <h1 className='text-base'>Data Label Count</h1>
          <h1 className='text-base'>{labelCount}</h1>
        </div>
        <Title text='Assign labels' />
        {/* display datalabel count */}
        {/* check len of combinations */}
        {
          combinations && combinations.length > 0
            ? (
              <Label combination={combinations[index]} setIndex={setIndex} />
            )
            : (
              <div className='flex justify-center items-center h-full'>
                <h1 className='text-2xl'>Loading...</h1>
              </div>
            )
        }
      </section>
    </>
  )
}


function App() {
  return (
    <main className="
      font-mono flex flex-col md:flex-row
      w-full
      h-screen min-h-screen max-h-screen
    ">
      <AssignLabel />
      <CreatePrompt />
    </main>
  )
}

export default App
