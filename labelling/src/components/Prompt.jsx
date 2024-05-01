/* eslint-disable react/prop-types */
const Prompt = ({ id, text, type, category, subcategory }) => {
  const typeColor = type === 'suppress' ? 'text-red-800' : 'text-green-500';

  return (
    <div className='border-2 rounded border-slate-700 p-2'>
      <div>
        <span className={typeColor}>{type}</span>
        <span className='ml-2 text-gray-500 font-bold'>{category}</span>
        <span className='ml-2 text-gray-500 font-bold'>{subcategory}</span>
      </div>
      <div>#{id} {text}</div>
    </div>
  );
};

export default Prompt;
