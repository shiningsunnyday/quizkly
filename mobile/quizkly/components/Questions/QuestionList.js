import React from 'react';
import { Text, FlatList, View, Button, StyleSheet } from 'react-native';
import Question from './Question';

export class QuestionList extends React.Component {

  constructor(props) {
    super(props);
    console.log(this.props.list);
  }

  state = {
    showTrueColor: false,
  }

  changeColor = () => {
    console.log("I'm technically about to change!");
    this.setState({ showTrueColor: !this.state.showTrueColor });
    console.log(this.state.showTrueColor);
    console.log("How's that new show true color?")
  }

  render() {
    return (
      <View>
        <View style={styles.buttons}>
          <Button style={styles.button} onPress={() => console.log("")} title="Test me with this quiz" />
          <Button style={styles.button} onPress={this.changeColor} title="Just show me the answers" />
        </View>
        <FlatList
          showTrueColor={this.state.showTrueColor}
          data={this.props.list}
          renderItem={({item}) =>
            <Question item={item} showTrueColor={this.state.showTrueColor}/>
          }
        />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  buttons: {
    height: 40,
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    backgroundColor: '#aaa'
  },
})
